# transcription_service

subscribes to the sensor_service to be called whenever a fresh audio chunk
comes into the system. 

Needs to do a rolling buffer of the last 5 seconds of audio and 1 second into the future of the
chunk being predicted on.

Must have a 1 second latency requirement, so the chunk being predicted on must be no more than 1 second old.
But we need the 1 second after the chunk for proper context to be predicted properly. 

Also need to detect a "hey cleo" or "hey clio" or anything that sounds like that and 
send whatever is transcribed in the next 3 seconds of audio to the assistant_service, including the "hey cleo" part.

When a complete line of the transcription is ready, send it to the data_service
with the timestamp of when the audio was recorded so it can be used for recall later.

We will run the model using Amazon Transcribe Streaming. 
Credentials are already setup using awscli.


