# CrewAI Debate: Multi-LLM Moderated Debate System

A sophisticated debate system using CrewAI where three AI agents with different language models engage in a structured, moderated debate. The Moderator agent dynamically controls the debate length based on discussion quality.

## Features

- 🤖 **Three specialized agents** using different LLMs via OpenRouter
- 🎯 **Moderator-controlled rounds** - debate length adapts to discussion quality
- 📝 **Comprehensive logging** - technical logs and human-readable conversation history
- 🔄 **Dynamic context** - agents see full debate history for better arguments
- ⚙️ **Configurable** - easily switch models and adjust maximum rounds

## Agents

All agents are fully configurable via the `.env` file:

1. **Proponent** (Default: GPT-5.1 via OpenRouter)
   - Builds strong arguments for the topic
   - Configurable: Model, role, goal, and backstory

2. **Opponent** (Default: Claude Sonnet 4.5 via OpenRouter)
   - Critiques arguments and finds weaknesses
   - Configurable: Model, role, goal, and backstory

3. **Moderator** (Default: Gemini 2.5 Pro via OpenRouter)
   - Evaluates debate quality and provides final summary
   - Configurable: Model, role, goal, and backstory

All models use the `OpenAICompletion` provider via OpenRouter's OpenAI-compatible API.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Get an OpenRouter API key:**
   - Visit https://openrouter.ai/keys
   - Create an account and generate an API key

3. **Configure environment:**
   ```bash
   # Copy the template to create your .env file
   cp .env.template .env
   ```
   
   Then edit `.env` and set your API key:
   ```bash
   OPENROUTER_API_KEY=your_actual_api_key_here
   ```
   
   **All configuration options in `.env`:**
   
   - **Required:**
     - `OPENROUTER_API_KEY` - Your OpenRouter API key
   
   - **Optional (with defaults):**
     - `MAX_DEBATE_ROUNDS` - Maximum rounds (default: 5)
     - `PROPONENT_MODEL` - Model for Proponent (default: `openai/gpt-5.1`)
     - `OPPONENT_MODEL` - Model for Opponent (default: `anthropic/claude-sonnet-4.5`)
     - `MODERATOR_MODEL` - Model for Moderator (default: `google/gemini-2.5-pro`)
     - `PROPONENT_ROLE` - Agent role name (default: "Proponent")
     - `PROPONENT_GOAL` - Agent goal/objective
     - `PROPONENT_BACKSTORY` - Agent backstory/personality
     - `OPPONENT_ROLE` - Agent role name (default: "Opponent")
     - `OPPONENT_GOAL` - Agent goal/objective
     - `OPPONENT_BACKSTORY` - Agent backstory/personality
     - `MODERATOR_ROLE` - Agent role name (default: "Moderator")
     - `MODERATOR_GOAL` - Agent goal/objective
     - `MODERATOR_BACKSTORY` - Agent backstory/personality
   
   See `.env.template` for full configuration options and defaults.

## Usage

### Basic Usage

Run the debate script with a topic as an argument:
```bash
python debate_crew.py "Artificial intelligence should be regulated by governments"
```

Or run without arguments to be prompted for a topic:
```bash
python debate_crew.py
```

### Example Topics

- "Artificial intelligence should be regulated by governments"
- "Remote work is better than office work"
- "Social media platforms should be held responsible for user content"
- "Universal basic income is necessary in an AI-driven economy"
- "Individual freedom is more valuable than social harmony"
- "Climate change requires immediate government intervention"

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

### Default Models

The system uses OpenRouter to access different models via OpenAI-compatible API:
- **GPT-5.1** for the Proponent (via `openai/gpt-5.1`)
- **Claude Sonnet 4.5** for the Opponent (via `anthropic/claude-sonnet-4.5`)
- **Gemini 2.5 Pro** for the Moderator (via `google/gemini-2.5-pro`)

### Customization

To use different models, simply edit your `.env` file:

```bash
# Example: Use different models
PROPONENT_MODEL=openai/gpt-4o
OPPONENT_MODEL=anthropic/claude-3.5-sonnet
MODERATOR_MODEL=google/gemini-2.0-flash-thinking-exp
```

Check available models at https://openrouter.ai/models

**Note:** All models use the `OpenAICompletion` provider since OpenRouter provides an OpenAI-compatible API endpoint.

### Customizing Agent Behavior

You can also customize each agent's role, goal, and backstory in the `.env` file:

```bash
# Example: Customize the Proponent agent
PROPONENT_ROLE=Advocate
PROPONENT_GOAL=Present evidence-based arguments with scientific rigor
PROPONENT_BACKSTORY=You are a research scientist with expertise in...
```

This allows you to tailor the debate style without modifying code.

## Project Structure

```
.
├── debate_crew.py          # Main debate script
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── .env.template          # Environment template (copy to .env)
├── .env                   # Your environment variables (create from template)
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

