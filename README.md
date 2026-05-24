# Local-AI-assistant

(Warning: extremely invasive, has access to your entire system.)

This code is mostly AI-generated, so it's probably extremely buggy and crappy, but I didn't wanna write this myself, but the actual interesting things this is also an open source project because it's AI-generated, and I don't think it should be closed-sourced if it's AI-generated. I should add this entire project will be available to all to run offline. That's the main objective of this: to have all of this run also offline. 

- Has access to all of your files and can run any command on your machine and can read everything
- It has a folder that is labeled as a watch folder. It will reference this if you ever mention anything about files, unless you directly ask it not to or to look at something outside of that file.
- Has access to see all of your hardware and current stuff, like basically Task Manager
- Work in progress again, AI-generated code, so probably pretty buggy.
- Everything is being run locally on your device, you should probably have a decent GPU.
- Just an AI assistant.

Requirements 
A good GPU (I'm using a RTX 4060. You don't necessarily need a GPU this high, but it can definitely result in slower response times.)
 Ollama with,
 1. For Speech-to-text	"faster-whisper small.en"
 2. For simple conversation "qwen2.5:1.5b"
 3. For normal standard commands or simple tasks. "qwen2.5:7b"
 4. For in-depth reasoning. "deepseek-r1:7b"
 5. For searching Embeddings "nomic-embed-text"
 6. For TTS "Kokoro-82M"
 7. For vision screen sharing to the model "Auto-detected from Ollama"
 8. 
