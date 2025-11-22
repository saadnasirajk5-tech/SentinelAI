# SentinelAI: The Reliable Agent Orchestrator       

A next-gen open-source framework that turns multiple AI models into honest, efficient, and self-correcting agents.
Unlike existing tools (CrewAI, AutoGen, LangGraph), SentinelAI:

forces agents to prove their answers
prevents infinite loops
tracks cost and switches to cheaper models
stores long-term memory
gives consistent results
works on laptop + cloud

Goal: Build the “Linux of AI Agents” — stable, cheap, and production-ready.




What each folder is for? 
1️ agents/

This is where each AI agent lives.
Examples:
researcher → searches & gathers info
writer → writes report
analyst → checks logic / summarization
Each agent = a class with:
a goal
a method like .run(input)
access to model (Phi, Llama, etc.)


2️ boss/

This is your "brain".
Includes:
supervisor → checks agent work
evaluator → gives confidence score, finds lies
retry logic
loop breaker
cost limiter
.

3️ memory/

Stores:
conversation
tool results
agent outputs
summaries
Uses ChromaDB or simple JSON at first.


4️ utils/

Reusable tools:
loading models
logging
tracking API cost
formatting
Start simple.

5️ tests/

6️ main.py

This is the “entry point”.
Example:
define task
pick which agents to call
call supervisor
print final result

















