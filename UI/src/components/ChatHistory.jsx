import React from 'react'

const ChatHistory = ({ chatHistory }) => {
  if (chatHistory.length === 0) {
    return null
  }

  return (
    <div className="space-y-4">
      {chatHistory.map((chat) => (
        <div key={chat.id} className="space-y-3">
          {/* User Message */}
          {chat.userMessage && (
            <div className="flex justify-end">
              <div className="bg-white text-black rounded-2xl rounded-tr-sm px-4 py-3 max-w-[85%] shadow-lg">
                <p className="text-sm">{chat.userMessage}</p>
              </div>
            </div>
          )}

          {/* Assistant Message */}
          {chat.assistantMessage && (
            <div className="flex justify-start">
              <div className="bg-neutral-900 border border-neutral-800 rounded-2xl rounded-tl-sm px-4 py-3 max-w-[85%]">
                <p className="text-sm text-neutral-200">{chat.assistantMessage}</p>
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  )
}

export default ChatHistory 