/* Shared API types matching the FastAPI Pydantic schemas. */

// ---- WebSocket protocol ----

export type ServerEventType =
  | 'status'
  | 'tool_progress'
  | 'tool_result'
  | 'response'
  | 'clarify'
  | 'session_info'
  | 'error'

export interface ServerEvent {
  type: ServerEventType
  // status
  status?: 'initializing' | 'processing' | 'done' | 'error'
  // tool_progress / tool_result
  tool_name?: string
  preview?: string
  result?: string
  success?: boolean
  // response
  content?: string
  api_calls?: number
  completed?: boolean
  interrupted?: boolean
  // clarify
  question?: string
  choices?: string[]
  // session_info
  session_id?: string
  message_count?: number
  resumed?: boolean
  // error
  message?: string
}

export type ClientMessageType = 'message' | 'interrupt' | 'clarify_response'

export interface ClientMessage {
  type: ClientMessageType
  content?: string
  session_id?: string
  answer?: string
}

// ---- REST models ----

export interface SessionSummary {
  id: string
  source: string
  model?: string
  started_at: number
  ended_at?: number
  end_reason?: string
  message_count: number
  tool_call_count: number
  input_tokens: number
  output_tokens: number
}

export interface SessionMessage {
  id: number
  session_id: string
  role: string
  content?: string
  tool_call_id?: string
  tool_calls?: unknown
  tool_name?: string
  timestamp: number
  token_count?: number
  finish_reason?: string
}

export interface ToolInfo {
  name: string
  description?: string
  toolset?: string
  available: boolean
  missing_env?: string[]
}

export interface ToolsetInfo {
  name: string
  tools: string[]
  description?: string
}

export interface MemoryData {
  target: string
  content: string
  entries: string[]
  usage: string
  entry_count: number
}

export interface ConfigData {
  config: Record<string, unknown>
}

export interface SkillCategory {
  name: string
  count: number
}

// ---- Chat UI types ----

export type ChatMessageRole = 'user' | 'assistant' | 'tool_progress' | 'system'

export interface ChatMessage {
  id: string
  role: ChatMessageRole
  content: string
  timestamp: number
  tool_name?: string
  tool_preview?: string
  isStreaming?: boolean
}
