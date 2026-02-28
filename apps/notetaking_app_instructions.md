# notetaking

This is another tool that inherits from tool_base.py

It can be executed in 2 cases

1. Hey Cleo, start notetaking
2. Hey Cleo, stop notetaking

Register that in `services/assistant/registry.py`
Keep it generic so they can say similar things to those statements and it will trigger the same tool.


When start notetaking is executed, it should simply log the timestamp of when it was executed
When stop notetaking is executed, it should log that timestamp as well and then query the 
database via `protos/data.proto` to get all the transcription data between those two timestamps
as well as the video clips between those two timestamps and then sends that all to a 
VLM to generate a text summary of everything that happened in that time period. 

Send these video clips + transcript to Claude Sonnet. 
Use the code in `services/assistant/bedrock.py` as reference for how to use Sonnet over bedrock
but have your own implementation in `apps/notetaking.py` that is separate from the assistant service.

Finally, update the data.proto to be able to accept these note summaries and store them in the 
sqlite database so that the frontend can query and show them later.

