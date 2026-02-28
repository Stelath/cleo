Transcription and tool calling service. 

Utilize https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3 to transcribe an audio stream and build
a transcript that is saved in memory. The transcript should be updated in real-time as new audio data is processed.

Whenever "hey cleo" is detected in the transcript, take the next 2 seconds of transcript and send it to the tool calling service. The tool calling service will analyze the text and determine if any tools need to be called based on the content of the transcript.
The tool calling service with then execute the most relevant tool based on the analysis of the transcript. 

The tooling call determining service will be configured with gRPC to allow for communication between the transcription service and the tool calling service. The transcription service will send the relevant transcript segment to the tool calling service via gRPC when the trigger phrase is detected.

In this directory write a Python file that implements the transcription service described above
as transcription.py. The file should include the necessary code to transcribe audio using the specified model, update the transcript in real-time, detect the trigger phrase "hey cleo", and send the relevant transcript segment to the tool calling service when the trigger phrase is detected.

The service will run locally. 

Write the transcription.py service which takes audio stream and output a transcript and can also call the tool calling service when the trigger phrase is detected.

It will interface with the rest of the system via protos/transcription.proto for gRPC communication.