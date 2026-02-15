package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"sync"
	"syscall"
	"time"
)

// ============================================================================
// Core Types
// ============================================================================

// Message represents a single message in the conversation.
type Message struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

// ToolCall represents a tool invocation requested by the LLM.
type ToolCall struct {
	ID       string        `json:"id"`
	Type     string        `json:"type"`
	Function *FunctionCall `json:"function,omitempty"`
}

// FunctionCall holds the function name and arguments JSON string.
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// LLMResponse is the parsed response from the LLM provider.
type LLMResponse struct {
	Content      string     `json:"content"`
	ToolCalls    []ToolCall `json:"tool_calls,omitempty"`
	FinishReason string     `json:"finish_reason"`
	Usage        *UsageInfo `json:"usage,omitempty"`
}

// UsageInfo tracks token usage for a single LLM call.
type UsageInfo struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ToolDefinition is the JSON schema sent to the LLM for tool discovery.
type ToolDefinition struct {
	Type     string          `json:"type"`
	Function ToolDefFunction `json:"function"`
}

// ToolDefFunction describes a tool's name, description, and parameter schema.
type ToolDefFunction struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters"`
}

// ============================================================================
// Tool Interface + ToolResult
// ============================================================================

// Tool is the interface that all tools (built-in, meta, and script) implement.
type Tool interface {
	Name() string
	Description() string
	Parameters() map[string]interface{}
	Execute(ctx context.Context, args map[string]interface{}) *ToolResult
}

// ToolResult holds the outcome of a tool execution.
type ToolResult struct {
	ForLLM  string
	ForUser string
	IsError bool
}

// ============================================================================
// Tool Registry
// ============================================================================

// ToolRegistry manages all registered tools and provides thread-safe access.
type ToolRegistry struct {
	mu       sync.RWMutex
	tools    map[string]Tool
	builtIns map[string]bool
}

// NewToolRegistry creates a new empty tool registry.
func NewToolRegistry() *ToolRegistry {
	return &ToolRegistry{
		tools:    make(map[string]Tool),
		builtIns: make(map[string]bool),
	}
}

// Register adds a tool to the registry. If builtIn is true, it is marked as
// protected against removal or overwrite.
func (r *ToolRegistry) Register(tool Tool, builtIn bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.tools[tool.Name()] = tool
	if builtIn {
		r.builtIns[tool.Name()] = true
	}
}

// Unregister removes a tool from the registry. Returns false if the tool does
// not exist or is built-in.
func (r *ToolRegistry) Unregister(name string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.builtIns[name] {
		return false
	}
	if _, ok := r.tools[name]; !ok {
		return false
	}
	delete(r.tools, name)
	return true
}

// Get retrieves a tool by name, or nil if not found.
func (r *ToolRegistry) Get(name string) Tool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.tools[name]
}

// Execute runs a tool by name with the given arguments.
func (r *ToolRegistry) Execute(ctx context.Context, name string, args map[string]interface{}) *ToolResult {
	tool := r.Get(name)
	if tool == nil {
		return &ToolResult{ForLLM: fmt.Sprintf("error: unknown tool '%s'", name), IsError: true}
	}
	return tool.Execute(ctx, args)
}

// Definitions returns the current list of tool definitions for the LLM.
// Called fresh each agent loop iteration so new tools appear immediately.
func (r *ToolRegistry) Definitions() []ToolDefinition {
	r.mu.RLock()
	defer r.mu.RUnlock()
	defs := make([]ToolDefinition, 0, len(r.tools))
	for _, tool := range r.tools {
		defs = append(defs, ToolDefinition{
			Type: "function",
			Function: ToolDefFunction{
				Name:        tool.Name(),
				Description: tool.Description(),
				Parameters:  tool.Parameters(),
			},
		})
	}
	return defs
}

// IsBuiltIn returns true if the named tool is a built-in tool.
func (r *ToolRegistry) IsBuiltIn(name string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.builtIns[name]
}

// ============================================================================
// Configuration
// ============================================================================

// Config holds all agent configuration values.
type Config struct {
	APIKey        string `json:"api_key"`
	APIBase       string `json:"api_base"`
	Model         string `json:"model"`
	MaxIterations int    `json:"max_iterations"`
	SessionWindow int    `json:"session_window"`
}

// configFile is the JSON structure of ~/.selfext/config.json.
type configFile struct {
	APIKey        string `json:"api_key,omitempty"`
	APIBase       string `json:"api_base,omitempty"`
	Model         string `json:"model,omitempty"`
	MaxIterations int    `json:"max_iterations,omitempty"`
	SessionWindow int    `json:"session_window,omitempty"`
}

// LoadConfig builds a Config from defaults, config file, and environment variables.
func LoadConfig() *Config {
	cfg := &Config{
		APIBase:       "https://api.openai.com/v1",
		Model:         "gpt-4o",
		MaxIterations: 20,
		SessionWindow: 50,
	}

	// Load from config file
	home, err := os.UserHomeDir()
	if err == nil {
		cfgPath := filepath.Join(home, ".selfext", "config.json")
		data, err := os.ReadFile(cfgPath)
		if err == nil {
			var fc configFile
			if json.Unmarshal(data, &fc) == nil {
				if fc.APIKey != "" {
					cfg.APIKey = fc.APIKey
				}
				if fc.APIBase != "" {
					cfg.APIBase = fc.APIBase
				}
				if fc.Model != "" {
					cfg.Model = fc.Model
				}
				if fc.MaxIterations > 0 {
					cfg.MaxIterations = fc.MaxIterations
				}
				if fc.SessionWindow > 0 {
					cfg.SessionWindow = fc.SessionWindow
				}
			}
		}
	}

	// Environment variables override everything
	if v := os.Getenv("SELFEXT_API_KEY"); v != "" {
		cfg.APIKey = v
	}
	if v := os.Getenv("SELFEXT_API_BASE"); v != "" {
		cfg.APIBase = v
	}
	if v := os.Getenv("SELFEXT_MODEL"); v != "" {
		cfg.Model = v
	}

	return cfg
}

// ============================================================================
// HTTP Provider
// ============================================================================

// HTTPProvider communicates with an OpenAI-compatible chat completions API.
type HTTPProvider struct {
	apiKey  string
	apiBase string
	model   string
	client  *http.Client
}

// NewHTTPProvider creates a new provider with the given configuration.
func NewHTTPProvider(cfg *Config) *HTTPProvider {
	return &HTTPProvider{
		apiKey:  cfg.APIKey,
		apiBase: strings.TrimRight(cfg.APIBase, "/"),
		model:   cfg.Model,
		client: &http.Client{
			Timeout: 120 * time.Second,
		},
	}
}

// chatRequest is the request body for the chat completions API.
type chatRequest struct {
	Model    string           `json:"model"`
	Messages []Message        `json:"messages"`
	Tools    []ToolDefinition `json:"tools,omitempty"`
}

// chatResponse is the top-level response from the chat completions API.
type chatResponse struct {
	Choices []chatChoice `json:"choices"`
	Usage   *UsageInfo   `json:"usage,omitempty"`
	Error   *apiError    `json:"error,omitempty"`
}

type chatChoice struct {
	Message      choiceMessage `json:"message"`
	FinishReason string        `json:"finish_reason"`
}

type choiceMessage struct {
	Role      string     `json:"role"`
	Content   string     `json:"content"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

type apiError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
}

// Complete sends the message history to the LLM and returns its response.
func (p *HTTPProvider) Complete(ctx context.Context, messages []Message, tools []ToolDefinition) (*LLMResponse, error) {
	reqBody := chatRequest{
		Model:    p.model,
		Messages: messages,
	}
	if len(tools) > 0 {
		reqBody.Tools = tools
	}

	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	url := p.apiBase + "/chat/completions"
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if p.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+p.apiKey)
	}

	resp, err := p.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("http request: %w", err)
	}
	defer resp.Body.Close()

	respBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API error (HTTP %d): %s", resp.StatusCode, string(respBytes))
	}

	var chatResp chatResponse
	if err := json.Unmarshal(respBytes, &chatResp); err != nil {
		return nil, fmt.Errorf("unmarshal response: %w", err)
	}

	if chatResp.Error != nil {
		return nil, fmt.Errorf("API error: %s (%s)", chatResp.Error.Message, chatResp.Error.Type)
	}

	if len(chatResp.Choices) == 0 {
		return nil, fmt.Errorf("no choices in response")
	}

	choice := chatResp.Choices[0]
	llmResp := &LLMResponse{
		Content:      choice.Message.Content,
		ToolCalls:    choice.Message.ToolCalls,
		FinishReason: choice.FinishReason,
		Usage:        chatResp.Usage,
	}
	return llmResp, nil
}

// ============================================================================
// Session (Sliding Window)
// ============================================================================

// Session manages the in-memory conversation history with a sliding window.
type Session struct {
	messages []Message
	window   int
}

// NewSession creates a new session with the given window size.
func NewSession(window int) *Session {
	return &Session{
		messages: make([]Message, 0, window),
		window:   window,
	}
}

// Add appends a message to the session, trimming old messages if needed.
func (s *Session) Add(msg Message) {
	s.messages = append(s.messages, msg)
	s.trim()
}

// Messages returns the current conversation history.
func (s *Session) Messages() []Message {
	return s.messages
}

// Clear resets the session.
func (s *Session) Clear() {
	s.messages = s.messages[:0]
}

// trim removes oldest messages (preserving system messages) to fit the window.
func (s *Session) trim() {
	if len(s.messages) <= s.window {
		return
	}
	// Always keep the first system message
	var sysMsg *Message
	if len(s.messages) > 0 && s.messages[0].Role == "system" {
		cp := s.messages[0]
		sysMsg = &cp
	}
	excess := len(s.messages) - s.window
	s.messages = s.messages[excess:]
	if sysMsg != nil && (len(s.messages) == 0 || s.messages[0].Role != "system") {
		s.messages = append([]Message{*sysMsg}, s.messages...)
	}
}

// ============================================================================
// Deny Patterns (Security)
// ============================================================================

var denyPatterns = []string{
	"rm -rf /",
	"rm -rf /*",
	"rm -rf ~",
	"rm -rf .",
	"mkfs.",
	"dd if=",
	":(){:|:&};:",
	"fork bomb",
	"shutdown",
	"reboot",
	"init 0",
	"init 6",
	"halt",
	"poweroff",
	"> /dev/sda",
	"chmod -R 777 /",
	"chown -R",
	":(){ :|:& };:",
}

// containsDenyPattern checks if text matches any deny pattern.
func containsDenyPattern(text string) (bool, string) {
	lower := strings.ToLower(text)
	for _, pattern := range denyPatterns {
		if strings.Contains(lower, strings.ToLower(pattern)) {
			return true, pattern
		}
	}
	return false, ""
}

// ============================================================================
// Built-in Tool: exec
// ============================================================================

type execTool struct{}

func (t *execTool) Name() string        { return "exec" }
func (t *execTool) Description() string { return "Execute a shell command and return its output." }
func (t *execTool) Parameters() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"command": map[string]interface{}{
				"type":        "string",
				"description": "The shell command to execute",
			},
		},
		"required": []string{"command"},
	}
}

func (t *execTool) Execute(ctx context.Context, args map[string]interface{}) *ToolResult {
	command, _ := args["command"].(string)
	if command == "" {
		return &ToolResult{ForLLM: "error: command is required", IsError: true}
	}

	if denied, pattern := containsDenyPattern(command); denied {
		return &ToolResult{
			ForLLM:  fmt.Sprintf("error: command denied - matches dangerous pattern: %s", pattern),
			IsError: true,
		}
	}

	shell := "sh"
	shellFlag := "-c"
	if runtime.GOOS == "windows" {
		shell = "cmd"
		shellFlag = "/c"
	}

	cmd := exec.CommandContext(ctx, shell, shellFlag, command)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err := cmd.Run()

	output := stdout.String()
	errOutput := stderr.String()
	if errOutput != "" {
		output += "\n[stderr]\n" + errOutput
	}

	output = truncateOutput(output, 10240)

	if err != nil {
		return &ToolResult{
			ForLLM:  fmt.Sprintf("command failed: %v\n%s", err, output),
			ForUser: fmt.Sprintf("exec: %s", command),
			IsError: true,
		}
	}

	return &ToolResult{
		ForLLM:  output,
		ForUser: fmt.Sprintf("exec: %s", command),
	}
}

// ============================================================================
// Built-in Tool: read_file
// ============================================================================

type readFileTool struct{}

func (t *readFileTool) Name() string        { return "read_file" }
func (t *readFileTool) Description() string { return "Read the contents of a file." }
func (t *readFileTool) Parameters() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"path": map[string]interface{}{
				"type":        "string",
				"description": "The path to the file to read",
			},
		},
		"required": []string{"path"},
	}
}

func (t *readFileTool) Execute(ctx context.Context, args map[string]interface{}) *ToolResult {
	path, _ := args["path"].(string)
	if path == "" {
		return &ToolResult{ForLLM: "error: path is required", IsError: true}
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return &ToolResult{ForLLM: fmt.Sprintf("error reading file: %v", err), IsError: true}
	}

	content := truncateOutput(string(data), 10240)
	return &ToolResult{
		ForLLM:  content,
		ForUser: fmt.Sprintf("read_file: %s", path),
	}
}

// ============================================================================
// Built-in Tool: write_file
// ============================================================================

type writeFileTool struct{}

func (t *writeFileTool) Name() string { return "write_file" }
func (t *writeFileTool) Description() string {
	return "Write content to a file, creating directories as needed."
}
func (t *writeFileTool) Parameters() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"path": map[string]interface{}{
				"type":        "string",
				"description": "The path to the file to write",
			},
			"content": map[string]interface{}{
				"type":        "string",
				"description": "The content to write to the file",
			},
		},
		"required": []string{"path", "content"},
	}
}

func (t *writeFileTool) Execute(ctx context.Context, args map[string]interface{}) *ToolResult {
	path, _ := args["path"].(string)
	content, _ := args["content"].(string)
	if path == "" {
		return &ToolResult{ForLLM: "error: path is required", IsError: true}
	}

	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return &ToolResult{ForLLM: fmt.Sprintf("error creating directories: %v", err), IsError: true}
	}

	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		return &ToolResult{ForLLM: fmt.Sprintf("error writing file: %v", err), IsError: true}
	}

	return &ToolResult{
		ForLLM:  fmt.Sprintf("wrote %d bytes to %s", len(content), path),
		ForUser: fmt.Sprintf("write_file: %s", path),
	}
}

// ============================================================================
// Built-in Tool: list_dir
// ============================================================================

type listDirTool struct{}

func (t *listDirTool) Name() string        { return "list_dir" }
func (t *listDirTool) Description() string { return "List files and directories in a given path." }
func (t *listDirTool) Parameters() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"path": map[string]interface{}{
				"type":        "string",
				"description": "The directory path to list",
			},
		},
		"required": []string{"path"},
	}
}

func (t *listDirTool) Execute(ctx context.Context, args map[string]interface{}) *ToolResult {
	path, _ := args["path"].(string)
	if path == "" {
		path = "."
	}

	entries, err := os.ReadDir(path)
	if err != nil {
		return &ToolResult{ForLLM: fmt.Sprintf("error listing directory: %v", err), IsError: true}
	}

	var sb strings.Builder
	for _, entry := range entries {
		if entry.IsDir() {
			sb.WriteString("DIR:  ")
		} else {
			sb.WriteString("FILE: ")
		}
		sb.WriteString(entry.Name())
		sb.WriteByte('\n')
	}

	return &ToolResult{
		ForLLM:  sb.String(),
		ForUser: fmt.Sprintf("list_dir: %s", path),
	}
}

// ============================================================================
// Built-in Tool: edit_file
// ============================================================================

type editFileTool struct{}

func (t *editFileTool) Name() string { return "edit_file" }
func (t *editFileTool) Description() string {
	return "Edit a file by replacing an exact string match. The old_string must appear exactly once in the file."
}
func (t *editFileTool) Parameters() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"path": map[string]interface{}{
				"type":        "string",
				"description": "The path to the file to edit",
			},
			"old_string": map[string]interface{}{
				"type":        "string",
				"description": "The exact string to find (must be unique in the file)",
			},
			"new_string": map[string]interface{}{
				"type":        "string",
				"description": "The string to replace it with",
			},
		},
		"required": []string{"path", "old_string", "new_string"},
	}
}

func (t *editFileTool) Execute(ctx context.Context, args map[string]interface{}) *ToolResult {
	path, _ := args["path"].(string)
	oldStr, _ := args["old_string"].(string)
	newStr, _ := args["new_string"].(string)

	if path == "" || oldStr == "" {
		return &ToolResult{ForLLM: "error: path and old_string are required", IsError: true}
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return &ToolResult{ForLLM: fmt.Sprintf("error reading file: %v", err), IsError: true}
	}

	content := string(data)
	count := strings.Count(content, oldStr)
	if count == 0 {
		return &ToolResult{ForLLM: "error: old_string not found in file", IsError: true}
	}
	if count > 1 {
		return &ToolResult{
			ForLLM:  fmt.Sprintf("error: old_string found %d times, must be unique", count),
			IsError: true,
		}
	}

	newContent := strings.Replace(content, oldStr, newStr, 1)
	if err := os.WriteFile(path, []byte(newContent), 0644); err != nil {
		return &ToolResult{ForLLM: fmt.Sprintf("error writing file: %v", err), IsError: true}
	}

	return &ToolResult{
		ForLLM:  fmt.Sprintf("edited %s: replaced 1 occurrence", path),
		ForUser: fmt.Sprintf("edit_file: %s", path),
	}
}

// ============================================================================
// ScriptTool (Dynamic / User-Created)
// ============================================================================

// ScriptTool is a tool backed by a script file on disk.
type ScriptTool struct {
	name        string
	description string
	params      map[string]interface{}
	scriptPath  string
	interpreter string
	timeout     time.Duration
}

func (t *ScriptTool) Name() string                       { return t.name }
func (t *ScriptTool) Description() string                { return t.description }
func (t *ScriptTool) Parameters() map[string]interface{} { return t.params }

func (t *ScriptTool) Execute(ctx context.Context, args map[string]interface{}) *ToolResult {
	timeout := t.timeout
	if timeout == 0 {
		timeout = 30 * time.Second
	}
	execCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	cmd := exec.CommandContext(execCtx, t.interpreter, t.scriptPath)

	// Build environment variables for parameters
	env := os.Environ()
	for key, val := range args {
		envKey := "TOOL_PARAM_" + strings.ToUpper(key)
		envVal := fmt.Sprintf("%v", val)
		env = append(env, envKey+"="+envVal)
	}

	// Also pass JSON on stdin and as env var
	argsJSON, err := json.Marshal(args)
	if err == nil {
		env = append(env, "TOOL_PARAMS_JSON="+string(argsJSON))
		cmd.Stdin = bytes.NewReader(argsJSON)
	}

	cmd.Env = env

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err = cmd.Run()

	output := stdout.String()
	errOutput := stderr.String()
	if errOutput != "" {
		output += "\n[stderr]\n" + errOutput
	}

	output = truncateOutput(output, 10240)

	if err != nil {
		return &ToolResult{
			ForLLM:  fmt.Sprintf("script error: %v\n%s", err, output),
			ForUser: fmt.Sprintf("script_tool(%s)", t.name),
			IsError: true,
		}
	}

	return &ToolResult{
		ForLLM:  output,
		ForUser: fmt.Sprintf("script_tool(%s)", t.name),
	}
}

// ============================================================================
// ToolStore (Persistence)
// ============================================================================

// toolManifest is the JSON structure of manifest.json.
type toolManifest struct {
	Tools []string `json:"tools"`
}

// toolDefinitionFile is the JSON structure of a per-tool definition file.
type toolDefinitionFile struct {
	Name           string                 `json:"name"`
	Description    string                 `json:"description"`
	Parameters     map[string]interface{} `json:"parameters"`
	Interpreter    string                 `json:"interpreter"`
	ScriptFile     string                 `json:"script_file"`
	TimeoutSeconds int                    `json:"timeout_seconds"`
}

// ToolStore handles persistence of custom tools to ~/.selfext/tools/.
type ToolStore struct {
	dir string
}

// NewToolStore creates a new ToolStore rooted at ~/.selfext/tools/.
func NewToolStore() (*ToolStore, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, fmt.Errorf("get home dir: %w", err)
	}
	dir := filepath.Join(home, ".selfext", "tools")
	store := &ToolStore{dir: dir}
	if err := store.EnsureDir(); err != nil {
		return nil, err
	}
	return store, nil
}

// EnsureDir creates the tools directory if it does not exist.
func (s *ToolStore) EnsureDir() error {
	return os.MkdirAll(s.dir, 0755)
}

// Save persists a ScriptTool to disk.
func (s *ToolStore) Save(tool *ScriptTool) error {
	if err := s.EnsureDir(); err != nil {
		return err
	}

	// Write the script file
	scriptPath := filepath.Join(s.dir, tool.name+".sh")
	scriptContent, err := os.ReadFile(tool.scriptPath)
	if err != nil {
		// If scriptPath IS already the target, read from there
		if tool.scriptPath != scriptPath {
			return fmt.Errorf("read script: %w", err)
		}
	} else if tool.scriptPath != scriptPath {
		if err := os.WriteFile(scriptPath, scriptContent, 0700); err != nil {
			return fmt.Errorf("write script: %w", err)
		}
		tool.scriptPath = scriptPath
	}

	// Write the definition file
	defFile := toolDefinitionFile{
		Name:           tool.name,
		Description:    tool.description,
		Parameters:     tool.params,
		Interpreter:    tool.interpreter,
		ScriptFile:     tool.name + ".sh",
		TimeoutSeconds: int(tool.timeout.Seconds()),
	}
	defBytes, err := json.MarshalIndent(defFile, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal definition: %w", err)
	}
	defPath := filepath.Join(s.dir, tool.name+".json")
	if err := os.WriteFile(defPath, defBytes, 0600); err != nil {
		return fmt.Errorf("write definition: %w", err)
	}

	// Update manifest
	return s.addToManifest(tool.name)
}

// SaveWithScript writes the script content to disk and then saves the tool definition.
func (s *ToolStore) SaveWithScript(tool *ScriptTool, scriptContent string) error {
	if err := s.EnsureDir(); err != nil {
		return err
	}

	scriptPath := filepath.Join(s.dir, tool.name+".sh")
	if err := os.WriteFile(scriptPath, []byte(scriptContent), 0700); err != nil {
		return fmt.Errorf("write script: %w", err)
	}
	tool.scriptPath = scriptPath

	// Write the definition file
	defFile := toolDefinitionFile{
		Name:           tool.name,
		Description:    tool.description,
		Parameters:     tool.params,
		Interpreter:    tool.interpreter,
		ScriptFile:     tool.name + ".sh",
		TimeoutSeconds: int(tool.timeout.Seconds()),
	}
	defBytes, err := json.MarshalIndent(defFile, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal definition: %w", err)
	}
	defPath := filepath.Join(s.dir, tool.name+".json")
	if err := os.WriteFile(defPath, defBytes, 0600); err != nil {
		return fmt.Errorf("write definition: %w", err)
	}

	return s.addToManifest(tool.name)
}

// Load reads a ScriptTool definition from disk.
func (s *ToolStore) Load(name string) (*ScriptTool, error) {
	defPath := filepath.Join(s.dir, name+".json")
	data, err := os.ReadFile(defPath)
	if err != nil {
		return nil, fmt.Errorf("read definition for '%s': %w", name, err)
	}

	var defFile toolDefinitionFile
	if err := json.Unmarshal(data, &defFile); err != nil {
		return nil, fmt.Errorf("parse definition for '%s': %w", name, err)
	}

	scriptPath := filepath.Join(s.dir, defFile.ScriptFile)
	if _, err := os.Stat(scriptPath); err != nil {
		return nil, fmt.Errorf("script file missing for '%s': %w", name, err)
	}

	timeout := time.Duration(defFile.TimeoutSeconds) * time.Second
	if timeout == 0 {
		timeout = 30 * time.Second
	}

	return &ScriptTool{
		name:        defFile.Name,
		description: defFile.Description,
		params:      defFile.Parameters,
		scriptPath:  scriptPath,
		interpreter: defFile.Interpreter,
		timeout:     timeout,
	}, nil
}

// Delete removes a tool's files and manifest entry.
func (s *ToolStore) Delete(name string) error {
	// Remove script file
	scriptPath := filepath.Join(s.dir, name+".sh")
	os.Remove(scriptPath)

	// Remove definition file
	defPath := filepath.Join(s.dir, name+".json")
	os.Remove(defPath)

	// Update manifest
	return s.removeFromManifest(name)
}

// LoadAll loads all tools listed in the manifest.
func (s *ToolStore) LoadAll() []*ScriptTool {
	manifest := s.readManifest()
	tools := make([]*ScriptTool, 0, len(manifest.Tools))
	for _, name := range manifest.Tools {
		tool, err := s.Load(name)
		if err != nil {
			fmt.Fprintf(os.Stderr, "warning: failed to load tool '%s': %v\n", name, err)
			continue
		}
		tools = append(tools, tool)
	}
	return tools
}

func (s *ToolStore) manifestPath() string {
	return filepath.Join(s.dir, "manifest.json")
}

func (s *ToolStore) readManifest() toolManifest {
	data, err := os.ReadFile(s.manifestPath())
	if err != nil {
		return toolManifest{Tools: []string{}}
	}
	var m toolManifest
	if json.Unmarshal(data, &m) != nil {
		return toolManifest{Tools: []string{}}
	}
	return m
}

func (s *ToolStore) writeManifest(m toolManifest) error {
	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(s.manifestPath(), data, 0600)
}

func (s *ToolStore) addToManifest(name string) error {
	m := s.readManifest()
	for _, t := range m.Tools {
		if t == name {
			return nil // already present
		}
	}
	m.Tools = append(m.Tools, name)
	return s.writeManifest(m)
}

func (s *ToolStore) removeFromManifest(name string) error {
	m := s.readManifest()
	filtered := make([]string, 0, len(m.Tools))
	for _, t := range m.Tools {
		if t != name {
			filtered = append(filtered, t)
		}
	}
	m.Tools = filtered
	return s.writeManifest(m)
}

// ============================================================================
// Meta-Tool: create_tool
// ============================================================================

var toolNameRegex = regexp.MustCompile(`^[a-z][a-z0-9_]{0,62}$`)
var validInterpreters = map[string]bool{
	"bash":    true,
	"sh":      true,
	"python3": true,
	"node":    true,
}

type createToolMeta struct {
	registry *ToolRegistry
	store    *ToolStore
}

func (t *createToolMeta) Name() string { return "create_tool" }
func (t *createToolMeta) Description() string {
	return "Create a new custom tool backed by a script. The tool will be immediately available for use and persisted to disk."
}
func (t *createToolMeta) Parameters() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"name": map[string]interface{}{
				"type":        "string",
				"description": "Tool name (lowercase, letters/numbers/underscores, max 63 chars)",
			},
			"description": map[string]interface{}{
				"type":        "string",
				"description": "Human-readable description of what the tool does",
			},
			"parameters": map[string]interface{}{
				"type":        "object",
				"description": "JSON Schema describing the tool's parameters",
			},
			"script": map[string]interface{}{
				"type":        "string",
				"description": "The script content to execute when the tool is called",
			},
			"interpreter": map[string]interface{}{
				"type":        "string",
				"description": "Script interpreter: bash, sh, python3, or node (default: bash)",
			},
			"timeout_seconds": map[string]interface{}{
				"type":        "integer",
				"description": "Execution timeout in seconds, 1-120 (default: 30)",
			},
		},
		"required": []string{"name", "description", "parameters", "script"},
	}
}

func (t *createToolMeta) Execute(ctx context.Context, args map[string]interface{}) *ToolResult {
	name, _ := args["name"].(string)
	desc, _ := args["description"].(string)
	script, _ := args["script"].(string)

	// Parse parameters - handle both map and raw JSON
	var params map[string]interface{}
	switch v := args["parameters"].(type) {
	case map[string]interface{}:
		params = v
	case string:
		if err := json.Unmarshal([]byte(v), &params); err != nil {
			return &ToolResult{ForLLM: "error: parameters must be a valid JSON Schema object", IsError: true}
		}
	default:
		return &ToolResult{ForLLM: "error: parameters must be a valid JSON Schema object", IsError: true}
	}

	interpreter, _ := args["interpreter"].(string)
	if interpreter == "" {
		interpreter = "bash"
	}

	timeoutSec := 30
	if v, ok := args["timeout_seconds"].(float64); ok {
		timeoutSec = int(v)
	}

	// Validation
	if name == "" {
		return &ToolResult{ForLLM: "error: name is required", IsError: true}
	}
	if !toolNameRegex.MatchString(name) {
		return &ToolResult{ForLLM: "error: name must match ^[a-z][a-z0-9_]{0,62}$", IsError: true}
	}
	if desc == "" {
		return &ToolResult{ForLLM: "error: description is required", IsError: true}
	}
	if script == "" {
		return &ToolResult{ForLLM: "error: script is required", IsError: true}
	}
	if t.registry.IsBuiltIn(name) {
		return &ToolResult{ForLLM: fmt.Sprintf("error: cannot overwrite built-in tool '%s'", name), IsError: true}
	}
	if !validInterpreters[interpreter] {
		return &ToolResult{
			ForLLM:  fmt.Sprintf("error: invalid interpreter '%s'. Must be one of: bash, sh, python3, node", interpreter),
			IsError: true,
		}
	}
	if timeoutSec < 1 {
		timeoutSec = 1
	}
	if timeoutSec > 120 {
		timeoutSec = 120
	}

	// Check script content for deny patterns
	if denied, pattern := containsDenyPattern(script); denied {
		return &ToolResult{
			ForLLM:  fmt.Sprintf("error: script contains dangerous pattern: %s", pattern),
			IsError: true,
		}
	}

	// Create the ScriptTool
	scriptTool := &ScriptTool{
		name:        name,
		description: desc,
		params:      params,
		interpreter: interpreter,
		timeout:     time.Duration(timeoutSec) * time.Second,
	}

	// Persist to disk
	if err := t.store.SaveWithScript(scriptTool, script); err != nil {
		return &ToolResult{ForLLM: fmt.Sprintf("error saving tool: %v", err), IsError: true}
	}

	// Register in live registry (overwrite if custom tool with same name exists)
	if existing := t.registry.Get(name); existing != nil {
		t.registry.Unregister(name)
	}
	t.registry.Register(scriptTool, false)

	return &ToolResult{
		ForLLM:  fmt.Sprintf("Tool '%s' created successfully. It is now available for use.", name),
		ForUser: fmt.Sprintf("created tool: %s", name),
	}
}

// ============================================================================
// Meta-Tool: list_custom_tools
// ============================================================================

type listCustomToolsMeta struct {
	registry *ToolRegistry
}

func (t *listCustomToolsMeta) Name() string        { return "list_custom_tools" }
func (t *listCustomToolsMeta) Description() string { return "List all registered custom tools." }
func (t *listCustomToolsMeta) Parameters() map[string]interface{} {
	return map[string]interface{}{
		"type":       "object",
		"properties": map[string]interface{}{},
	}
}

func (t *listCustomToolsMeta) Execute(ctx context.Context, args map[string]interface{}) *ToolResult {
	t.registry.mu.RLock()
	defer t.registry.mu.RUnlock()

	var sb strings.Builder
	count := 0
	for name, tool := range t.registry.tools {
		if t.registry.builtIns[name] {
			continue
		}
		st, ok := tool.(*ScriptTool)
		if !ok {
			continue
		}
		count++
		sb.WriteString(fmt.Sprintf("- %s (%s): %s\n", st.name, st.interpreter, st.description))
	}

	if count == 0 {
		return &ToolResult{ForLLM: "No custom tools are currently registered."}
	}

	return &ToolResult{
		ForLLM:  fmt.Sprintf("%d custom tool(s):\n%s", count, sb.String()),
		ForUser: "listed custom tools",
	}
}

// ============================================================================
// Meta-Tool: remove_tool
// ============================================================================

type removeToolMeta struct {
	registry *ToolRegistry
	store    *ToolStore
}

func (t *removeToolMeta) Name() string        { return "remove_tool" }
func (t *removeToolMeta) Description() string { return "Remove a custom tool by name." }
func (t *removeToolMeta) Parameters() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"name": map[string]interface{}{
				"type":        "string",
				"description": "The name of the custom tool to remove",
			},
		},
		"required": []string{"name"},
	}
}

func (t *removeToolMeta) Execute(ctx context.Context, args map[string]interface{}) *ToolResult {
	name, _ := args["name"].(string)
	if name == "" {
		return &ToolResult{ForLLM: "error: name is required", IsError: true}
	}

	if t.registry.IsBuiltIn(name) {
		return &ToolResult{ForLLM: fmt.Sprintf("error: cannot remove built-in tool '%s'", name), IsError: true}
	}

	if t.registry.Get(name) == nil {
		return &ToolResult{ForLLM: fmt.Sprintf("error: tool '%s' not found", name), IsError: true}
	}

	t.registry.Unregister(name)
	if err := t.store.Delete(name); err != nil {
		return &ToolResult{ForLLM: fmt.Sprintf("warning: tool unregistered but failed to delete files: %v", err), IsError: false}
	}

	return &ToolResult{
		ForLLM:  fmt.Sprintf("Tool '%s' removed successfully.", name),
		ForUser: fmt.Sprintf("removed tool: %s", name),
	}
}

// ============================================================================
// Meta-Tool: update_tool
// ============================================================================

type updateToolMeta struct {
	registry *ToolRegistry
	store    *ToolStore
}

func (t *updateToolMeta) Name() string { return "update_tool" }
func (t *updateToolMeta) Description() string {
	return "Update an existing custom tool. Only provide the fields you want to change."
}
func (t *updateToolMeta) Parameters() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"name": map[string]interface{}{
				"type":        "string",
				"description": "The name of the custom tool to update",
			},
			"description": map[string]interface{}{
				"type":        "string",
				"description": "New description for the tool",
			},
			"parameters": map[string]interface{}{
				"type":        "object",
				"description": "New JSON Schema for the tool's parameters",
			},
			"script": map[string]interface{}{
				"type":        "string",
				"description": "New script content",
			},
			"interpreter": map[string]interface{}{
				"type":        "string",
				"description": "New interpreter: bash, sh, python3, or node",
			},
			"timeout_seconds": map[string]interface{}{
				"type":        "integer",
				"description": "New execution timeout in seconds, 1-120",
			},
		},
		"required": []string{"name"},
	}
}

func (t *updateToolMeta) Execute(ctx context.Context, args map[string]interface{}) *ToolResult {
	name, _ := args["name"].(string)
	if name == "" {
		return &ToolResult{ForLLM: "error: name is required", IsError: true}
	}

	if t.registry.IsBuiltIn(name) {
		return &ToolResult{ForLLM: fmt.Sprintf("error: cannot update built-in tool '%s'", name), IsError: true}
	}

	existing := t.registry.Get(name)
	if existing == nil {
		return &ToolResult{ForLLM: fmt.Sprintf("error: tool '%s' not found", name), IsError: true}
	}

	st, ok := existing.(*ScriptTool)
	if !ok {
		return &ToolResult{ForLLM: fmt.Sprintf("error: tool '%s' is not a custom tool", name), IsError: true}
	}

	// Apply updates
	var newScript string
	needScriptWrite := false

	if desc, ok := args["description"].(string); ok && desc != "" {
		st.description = desc
	}
	if params, ok := args["parameters"].(map[string]interface{}); ok {
		st.params = params
	}
	if paramsStr, ok := args["parameters"].(string); ok {
		var params map[string]interface{}
		if err := json.Unmarshal([]byte(paramsStr), &params); err == nil {
			st.params = params
		}
	}
	if script, ok := args["script"].(string); ok && script != "" {
		if denied, pattern := containsDenyPattern(script); denied {
			return &ToolResult{
				ForLLM:  fmt.Sprintf("error: script contains dangerous pattern: %s", pattern),
				IsError: true,
			}
		}
		newScript = script
		needScriptWrite = true
	}
	if interp, ok := args["interpreter"].(string); ok && interp != "" {
		if !validInterpreters[interp] {
			return &ToolResult{
				ForLLM:  fmt.Sprintf("error: invalid interpreter '%s'", interp),
				IsError: true,
			}
		}
		st.interpreter = interp
	}
	if v, ok := args["timeout_seconds"].(float64); ok {
		sec := int(v)
		if sec < 1 {
			sec = 1
		}
		if sec > 120 {
			sec = 120
		}
		st.timeout = time.Duration(sec) * time.Second
	}

	// Persist
	var err error
	if needScriptWrite {
		err = t.store.SaveWithScript(st, newScript)
	} else {
		err = t.store.Save(st)
	}
	if err != nil {
		return &ToolResult{ForLLM: fmt.Sprintf("error saving tool: %v", err), IsError: true}
	}

	// Re-register
	t.registry.Unregister(name)
	t.registry.Register(st, false)

	return &ToolResult{
		ForLLM:  fmt.Sprintf("Tool '%s' updated successfully.", name),
		ForUser: fmt.Sprintf("updated tool: %s", name),
	}
}

// ============================================================================
// System Prompt
// ============================================================================

func buildSystemPrompt(registry *ToolRegistry) string {
	var sb strings.Builder
	sb.WriteString("You are the Self-Extending Agent, an AI assistant with the unique ability to create your own tools at runtime.\n\n")
	sb.WriteString("You have built-in tools for file operations and shell commands. You also have meta-tools that allow you to:\n")
	sb.WriteString("- create_tool: Create new custom tools backed by scripts (bash, sh, python3, node)\n")
	sb.WriteString("- list_custom_tools: See all custom tools you've created\n")
	sb.WriteString("- update_tool: Modify existing custom tools\n")
	sb.WriteString("- remove_tool: Delete custom tools you no longer need\n\n")
	sb.WriteString("When you encounter a task that would benefit from a reusable tool, create one! Custom tools:\n")
	sb.WriteString("- Receive parameters as TOOL_PARAM_<NAME> environment variables (uppercased)\n")
	sb.WriteString("- Also receive all parameters as JSON on stdin (TOOL_PARAMS_JSON env var)\n")
	sb.WriteString("- Return output via stdout (stderr is captured for context)\n")
	sb.WriteString("- Are persisted to disk and survive across sessions\n\n")

	sb.WriteString("Currently available tools:\n")
	registry.mu.RLock()
	for name, tool := range registry.tools {
		prefix := "  "
		if registry.builtIns[name] {
			prefix = "  [built-in] "
		} else if _, ok := tool.(*ScriptTool); ok {
			prefix = "  [custom]   "
		} else {
			prefix = "  [meta]     "
		}
		sb.WriteString(fmt.Sprintf("%s%s: %s\n", prefix, name, tool.Description()))
	}
	registry.mu.RUnlock()

	sb.WriteString("\nAlways think step-by-step. Use tools when helpful. Create custom tools for tasks you might repeat.")
	return sb.String()
}

// ============================================================================
// Agent
// ============================================================================

// Agent orchestrates the LLM, session, and tool registry.
type Agent struct {
	provider      *HTTPProvider
	session       *Session
	registry      *ToolRegistry
	maxIterations int
}

// NewAgent creates a new agent.
func NewAgent(provider *HTTPProvider, session *Session, registry *ToolRegistry, maxIter int) *Agent {
	return &Agent{
		provider:      provider,
		session:       session,
		registry:      registry,
		maxIterations: maxIter,
	}
}

// Run sends the user message through the agent loop, returning the final text response.
func (a *Agent) Run(ctx context.Context, userMessage string) (string, error) {
	// Add system prompt
	if len(a.session.Messages()) == 0 || a.session.Messages()[0].Role != "system" {
		sysPrompt := buildSystemPrompt(a.registry)
		a.session.messages = append([]Message{{Role: "system", Content: sysPrompt}}, a.session.messages...)
	} else {
		// Update system prompt to reflect current tools
		a.session.messages[0].Content = buildSystemPrompt(a.registry)
	}

	// Add user message
	a.session.Add(Message{Role: "user", Content: userMessage})

	for i := 0; i < a.maxIterations; i++ {
		// Get fresh tool definitions each iteration
		toolDefs := a.registry.Definitions()

		llmResp, err := a.provider.Complete(ctx, a.session.Messages(), toolDefs)
		if err != nil {
			return "", fmt.Errorf("LLM error: %w", err)
		}

		// If there are no tool calls, this is the final response
		if len(llmResp.ToolCalls) == 0 {
			if llmResp.Content != "" {
				a.session.Add(Message{Role: "assistant", Content: llmResp.Content})
			}
			return llmResp.Content, nil
		}

		// Add assistant message with tool calls
		assistantMsg := Message{
			Role:      "assistant",
			Content:   llmResp.Content,
			ToolCalls: llmResp.ToolCalls,
		}
		a.session.Add(assistantMsg)

		// Execute each tool call
		for _, tc := range llmResp.ToolCalls {
			if tc.Function == nil {
				continue
			}

			// Parse arguments
			var args map[string]interface{}
			if tc.Function.Arguments != "" {
				if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
					args = make(map[string]interface{})
				}
			} else {
				args = make(map[string]interface{})
			}

			// Show user what's happening
			result := a.registry.Execute(ctx, tc.Function.Name, args)
			if result.ForUser != "" {
				fmt.Fprintf(os.Stderr, "  [tool] %s\n", result.ForUser)
			}

			// Add tool result message
			toolMsg := Message{
				Role:       "tool",
				Content:    result.ForLLM,
				ToolCallID: tc.ID,
			}
			a.session.Add(toolMsg)
		}
	}

	return "Maximum iterations reached. Please try rephrasing your request.", nil
}

// ============================================================================
// Utility Functions
// ============================================================================

// truncateOutput trims output to maxLen bytes, appending a truncation notice.
func truncateOutput(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "\n... [output truncated]"
}

// ============================================================================
// CLI / REPL
// ============================================================================

func printHelp() {
	fmt.Println("Self-Extending Agent - an AI agent that creates its own tools at runtime")
	fmt.Println()
	fmt.Println("Commands:")
	fmt.Println("  /help    - Show this help message")
	fmt.Println("  /tools   - List all registered tools")
	fmt.Println("  /clear   - Clear conversation history")
	fmt.Println("  /quit    - Exit the agent")
	fmt.Println("  /exit    - Exit the agent")
	fmt.Println()
}

func printTools(registry *ToolRegistry) {
	registry.mu.RLock()
	defer registry.mu.RUnlock()

	fmt.Println("Registered tools:")
	fmt.Println()

	// Built-in tools
	fmt.Println("  Built-in:")
	for name, tool := range registry.tools {
		if registry.builtIns[name] {
			fmt.Printf("    %-20s %s\n", name, tool.Description())
		}
	}

	// Meta tools
	fmt.Println()
	fmt.Println("  Meta:")
	metaNames := []string{"create_tool", "list_custom_tools", "remove_tool", "update_tool"}
	for _, mn := range metaNames {
		if tool, ok := registry.tools[mn]; ok {
			fmt.Printf("    %-20s %s\n", mn, tool.Description())
		}
	}

	// Custom tools
	hasCustom := false
	for name, tool := range registry.tools {
		if !registry.builtIns[name] {
			if _, ok := tool.(*ScriptTool); ok {
				if !hasCustom {
					fmt.Println()
					fmt.Println("  Custom:")
					hasCustom = true
				}
				st := tool.(*ScriptTool)
				fmt.Printf("    %-20s [%s] %s\n", name, st.interpreter, st.description)
			}
		}
	}
	if !hasCustom {
		fmt.Println()
		fmt.Println("  Custom: (none)")
	}
	fmt.Println()
}

func runREPL(agent *Agent, registry *ToolRegistry) {
	scanner := bufio.NewScanner(os.Stdin)
	// Increase scanner buffer for long inputs
	scanner.Buffer(make([]byte, 0, 1024*1024), 1024*1024)

	fmt.Println("Self-Extending Agent")
	fmt.Println("Type /help for commands, /quit to exit")
	fmt.Println()

	// Set up signal handling for Ctrl+C
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT)

	for {
		fmt.Print("selfext> ")
		if !scanner.Scan() {
			break
		}

		input := strings.TrimSpace(scanner.Text())
		if input == "" {
			continue
		}

		// Handle special commands
		switch input {
		case "/quit", "/exit":
			fmt.Println("Goodbye.")
			return
		case "/help":
			printHelp()
			continue
		case "/clear":
			agent.session.Clear()
			fmt.Println("Conversation cleared.")
			continue
		case "/tools":
			printTools(registry)
			continue
		}

		// Create a cancellable context for this request
		ctx, cancel := context.WithCancel(context.Background())

		// Handle Ctrl+C to cancel current operation
		go func() {
			select {
			case <-sigChan:
				fmt.Fprintf(os.Stderr, "\n  [cancelled]\n")
				cancel()
			case <-ctx.Done():
			}
		}()

		response, err := agent.Run(ctx, input)
		cancel()

		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			continue
		}

		if response != "" {
			fmt.Println()
			fmt.Println(response)
			fmt.Println()
		}
	}

	if err := scanner.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "Input error: %v\n", err)
	}
}

func runOneShot(agent *Agent, message string) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Handle Ctrl+C
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT)
	go func() {
		<-sigChan
		cancel()
	}()

	response, err := agent.Run(ctx, message)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	if response != "" {
		fmt.Println(response)
	}
}

// ============================================================================
// Main
// ============================================================================

func main() {
	// Parse flags
	messageFlag := flag.String("m", "", "One-shot mode: send a single message")
	flag.Parse()

	// 1. Load config
	cfg := LoadConfig()
	if cfg.APIKey == "" {
		fmt.Fprintln(os.Stderr, "Error: API key not set. Set SELFEXT_API_KEY environment variable or configure ~/.selfext/config.json")
		os.Exit(1)
	}

	// 2. Create registry, register built-in tools
	registry := NewToolRegistry()
	registry.Register(&execTool{}, true)
	registry.Register(&readFileTool{}, true)
	registry.Register(&writeFileTool{}, true)
	registry.Register(&listDirTool{}, true)
	registry.Register(&editFileTool{}, true)

	// 3. Create tool store
	store, err := NewToolStore()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating tool store: %v\n", err)
		os.Exit(1)
	}

	// 4. Load persisted custom tools from ~/.selfext/tools/
	customTools := store.LoadAll()

	// 5. Register loaded custom tools
	for _, ct := range customTools {
		registry.Register(ct, false)
		fmt.Fprintf(os.Stderr, "  loaded custom tool: %s\n", ct.name)
	}

	// 6. Register meta-tools
	registry.Register(&createToolMeta{registry: registry, store: store}, true)
	registry.Register(&listCustomToolsMeta{registry: registry}, true)
	registry.Register(&removeToolMeta{registry: registry, store: store}, true)
	registry.Register(&updateToolMeta{registry: registry, store: store}, true)

	// 7. Create agent
	provider := NewHTTPProvider(cfg)
	session := NewSession(cfg.SessionWindow)
	agent := NewAgent(provider, session, registry, cfg.MaxIterations)

	// 8. Dispatch to REPL or one-shot
	if *messageFlag != "" {
		runOneShot(agent, *messageFlag)
	} else {
		runREPL(agent, registry)
	}
}
