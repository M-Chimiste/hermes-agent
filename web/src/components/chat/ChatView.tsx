import { useChat } from '@/hooks/useChat'
import { Header } from '@/components/layout/Header'
import { MessageList } from './MessageList'
import { ChatInput } from './ChatInput'
import { ClarifyDialog } from './ClarifyDialog'

export function ChatView() {
  const {
    messages,
    wsState,
    isProcessing,
    sessionId,
    clarifyPrompt,
    sendMessage,
    interrupt,
    respondToClarify,
  } = useChat()

  const isDisconnected = wsState !== 'connected'

  return (
    <div className="flex flex-col h-full">
      <Header wsState={wsState} sessionId={sessionId} />
      <MessageList messages={messages} isProcessing={isProcessing} />
      <ChatInput
        onSend={sendMessage}
        onInterrupt={interrupt}
        isProcessing={isProcessing}
        disabled={isDisconnected}
      />
      {clarifyPrompt && (
        <ClarifyDialog
          question={clarifyPrompt.question}
          choices={clarifyPrompt.choices}
          onRespond={respondToClarify}
        />
      )}
    </div>
  )
}
