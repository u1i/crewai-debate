# CrewAI Debate: Multi-LLM Moderated Debate System

A sophisticated debate system using CrewAI where three AI agents with different language models engage in a structured, moderated debate. The Moderator agent dynamically controls the debate length based on discussion quality.

## Features

- 🤖 **Three specialized agents** using different LLMs via OpenRouter
- 🎯 **Moderator-controlled rounds** - debate length adapts to discussion quality
- 📝 **Comprehensive logging** - technical logs and human-readable conversation history
- 🔄 **Dynamic context** - agents see full debate history for better arguments
- ⚙️ **Configurable** - easily switch models and adjust maximum rounds

## Agents

1. **Proponent** (GPT-5.1 via OpenRouter)
   - Role: Arguer
   - Goal: Build the strongest possible argument for the topic
   - Model: `openai/gpt-5.1`
   - Uses: OpenAICompletion provider

2. **Opponent** (Claude Sonnet 4.5 via OpenRouter)
   - Role: Critic
   - Goal: Find every flaw, logical fallacy, and weak point in the Proponent's argument
   - Model: `anthropic/claude-sonnet-4.5`
   - Uses: OpenAICompletion provider (OpenRouter-compatible)

3. **Moderator** (Gemini 2.5 Pro via OpenRouter)
   - Role: Synthesizer & Judge
   - Goal: Evaluate debate quality and decide when to end, then provide final summary
   - Model: `google/gemini-2.5-pro`
   - Uses: OpenAICompletion provider (OpenRouter-compatible)

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Get an OpenRouter API key:**
   - Visit https://openrouter.ai/keys
   - Create an account and generate an API key

3. **Configure environment:**
   Create a `.env` file with:
   ```
   OPENROUTER_API_KEY=your_actual_api_key_here
   MAX_DEBATE_ROUNDS=5
   ```
   
   - `OPENROUTER_API_KEY`: Your OpenRouter API key (required)
   - `MAX_DEBATE_ROUNDS`: Maximum number of debate rounds allowed (default: 5, optional)

## Usage

### Basic Usage

Run the debate script with a topic as an argument:
```bash
python debate_crew.py "Should artificial intelligence be regulated by governments?"
```

Or run without arguments to be prompted for a topic:
```bash
python debate_crew.py
```

### Example Topics

- "Should artificial intelligence be regulated by governments?"
- "Is remote work better than office work?"
- "Should social media platforms be held responsible for user content?"
- "Is universal basic income necessary in an AI-driven economy?"

## How It Works

The debate is **moderator-controlled** and consists of dynamic rounds:

1. **Round 1**: 
   - Proponent makes an opening argument
   - Opponent critiques and presents counterarguments (with full context)
   - **Moderator evaluates**: Should the debate continue? (CONTINUE or DONE)

2. **Round 2+** (if Moderator says "CONTINUE"):
   - Proponent defends and rebuts (sees all previous statements)
   - Opponent responds with additional critiques (sees all previous statements)
   - **Moderator evaluates again**: Continue or done?

3. **This continues** until:
   - Moderator decides the debate has reached sufficient depth ("DONE"), OR
   - Maximum rounds (from `.env`) is reached

4. **Final**: Moderator synthesizes the entire debate into a balanced summary

**Key Features:**
- The **Moderator decides** when the debate is complete (not a fixed number of rounds)
- Maximum rounds limit is set in `.env` (`MAX_DEBATE_ROUNDS`, default: 5)
- Each round allows deeper engagement with arguments
- The debate can end early if the Moderator determines sufficient depth has been reached
- **Full context**: All agents see the complete debate history for better arguments

## Logging

The system generates two types of logs:

1. **Technical Log** (`logs/debate_TIMESTAMP_TOPIC.log`)
   - Detailed execution logs with timestamps
   - Task creation and execution details
   - System-level information

2. **Conversation Log** (`logs/conversation_TIMESTAMP_TOPIC.md`)
   - Human-readable markdown format
   - Complete conversation history
   - Shows inputs (prompts/context) and outputs (responses) for each agent
   - Perfect for reviewing the debate flow

Both logs are created automatically in the `logs/` directory.

## Model Configuration

The example uses OpenRouter to access different models via OpenAI-compatible API:
- **GPT-5.1** for the Proponent (via `openai/gpt-5.1`)
- **Claude Sonnet 4.5** for the Opponent (via `anthropic/claude-sonnet-4.5`)
- **Gemini 2.5 Pro** for the Moderator (via `google/gemini-2.5-pro`)

### Customization

To use different models, modify the `model` parameter in each agent's LLM configuration in `debate_crew.py`:

```python
# In create_proponent_agent():
proponent_llm = OpenAICompletion(
    model="openai/gpt-4o",  # Change model here
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
    ...
)
```

Check available models at https://openrouter.ai/models

**Note:** All models use the `OpenAICompletion` provider since OpenRouter provides an OpenAI-compatible API endpoint.

## Project Structure

```
.
├── debate_crew.py          # Main debate script
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── .env                   # Environment variables (create this)
├── .gitignore            # Git ignore rules
└── logs/                  # Generated log files (auto-created)
    ├── debate_*.log       # Technical logs
    └── conversation_*.md  # Conversation history logs
```

## Requirements

- Python 3.8+
- OpenRouter API key
- Internet connection for API calls

## Dependencies

- `crewai` - Multi-agent framework
- `langchain-openai` - LLM integration
- `python-dotenv` - Environment variable management

## License

This is an example project. Feel free to use and modify as needed.

## Contributing

Feel free to submit issues or pull requests if you have improvements or find bugs!

