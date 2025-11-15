"""
CrewAI Example: Role-Based Debate with Multiple LLMs via OpenRouter

This example demonstrates three agents engaging in a moderator-controlled debate:
- Proponent (GPT-4o): Builds strong arguments and defends position
- Opponent (Claude Sonnet 3.5): Critiques arguments and presents counterarguments
- Moderator (Gemini 2.0 Flash): Decides when debate is complete, then synthesizes final summary

The debate flow:
1. Proponent makes an opening argument
2. Opponent critiques and counters
3. Moderator evaluates: Should debate continue? (CONTINUE or DONE)
4. If CONTINUE: Proponent defends, Opponent critiques, Moderator evaluates again
5. Continues until Moderator says DONE or MAX_ROUNDS (from .env) is reached
6. Moderator provides final comprehensive summary

Note: You can modify the models to use GPT-5 or Gemini 2.5 Pro when available.
"""

import os
import logging
from datetime import datetime
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.llms.providers.openai.completion import OpenAICompletion

# Load environment variables
load_dotenv()

# OpenRouter API configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("Please set OPENROUTER_API_KEY in your .env file")

# Maximum debate rounds (from .env, default: 5)
MAX_ROUNDS = int(os.getenv("MAX_DEBATE_ROUNDS", "5"))

# OpenRouter base URL
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Setup logging
def setup_logging(topic: str):
    """Setup file logging for the debate session.
    
    Returns:
        tuple: (logger, log_filename, conversation_log_filename)
    """
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Create log filename with timestamp and sanitized topic
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = "".join(c for c in topic[:50] if c.isalnum() or c in (' ', '-', '_')).strip().replace(' ', '_')
    log_filename = f"logs/debate_{timestamp}_{safe_topic}.log"
    conversation_log_filename = f"logs/conversation_{timestamp}_{safe_topic}.md"
    
    # Configure logging
    logger = logging.getLogger("debate_logger")
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    logger.handlers = []
    
    # File handler for detailed logging
    file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Console handler for summary
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    
    # Formatter
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(detailed_formatter)
    console_handler.setFormatter(detailed_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Initialize conversation log file
    with open(conversation_log_filename, 'w', encoding='utf-8') as f:
        f.write(f"# Debate Conversation History\n\n")
        f.write(f"**Topic:** {topic}\n\n")
        f.write(f"**Started:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Maximum Rounds:** {MAX_ROUNDS}\n\n")
        f.write("---\n\n")
    
    return logger, log_filename, conversation_log_filename


def write_conversation_entry(log_filename: str, round_num: int, agent_role: str, 
                            input_text: str, output_text: str, is_evaluation: bool = False):
    """Write a conversation entry to the human-readable log file.
    
    Args:
        log_filename: Path to the conversation log file
        round_num: Round number
        agent_role: Role of the agent (Proponent, Opponent, Moderator)
        input_text: Input/prompt given to the agent
        output_text: Output/response from the agent
        is_evaluation: Whether this is a moderator evaluation (CONTINUE/DONE)
    """
    with open(log_filename, 'a', encoding='utf-8') as f:
        if is_evaluation:
            f.write(f"## Round {round_num} - Moderator Evaluation\n\n")
        else:
            f.write(f"## Round {round_num} - {agent_role}\n\n")
        
        f.write(f"### Input (Prompt/Context)\n\n")
        f.write(f"```\n{input_text}\n```\n\n")
        
        f.write(f"### Output (Response)\n\n")
        if is_evaluation:
            f.write(f"**Decision:** `{output_text}`\n\n")
        else:
            f.write(f"{output_text}\n\n")
        
        f.write("---\n\n")


def create_proponent_agent():
    """Create the Proponent agent using GPT-5.1 via OpenRouter.
    
    To use GPT-5 when available, change the model to: "openai/gpt-5"
    """
    # Use native OpenAI provider directly with OpenRouter
    proponent_llm = OpenAICompletion(
        model="openai/gpt-5.1",
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        default_headers={
            "HTTP-Referer": "https://github.com/your-repo",
            "X-Title": "CrewAI Debate Example"
        }
    )
    
    return Agent(
        role="Proponent",
        goal="Build the strongest possible argument for the given topic. Use logical reasoning, evidence, and persuasive techniques to construct a compelling case.",
        backstory="""You are a skilled debater and advocate with expertise in constructing 
        well-reasoned arguments. Your goal is to present the most convincing case possible, 
        using facts, logic, and rhetorical techniques to support your position.""",
        verbose=True,
        allow_delegation=False,
        llm=proponent_llm
    )


def create_opponent_agent():
    """Create the Opponent agent using Claude Sonnet 4.5 via OpenRouter."""
    # Use OpenAI provider (OpenRouter is OpenAI-compatible) for Claude model
    opponent_llm = OpenAICompletion(
        model="anthropic/claude-sonnet-4.5",
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        default_headers={
            "HTTP-Referer": "https://github.com/your-repo",
            "X-Title": "CrewAI Debate Example"
        }
    )
    
    return Agent(
        role="Opponent",
        goal="Find every flaw, logical fallacy, and weak point in the Proponent's argument. Challenge assumptions, identify gaps, and expose weaknesses.",
        backstory="""You are a critical thinker and skilled critic with a keen eye for 
        identifying logical fallacies, weak reasoning, and unsupported claims. Your role 
        is to rigorously examine arguments and expose their vulnerabilities.""",
        verbose=True,
        allow_delegation=False,
        llm=opponent_llm
    )


def create_moderator_agent():
    """Create the Moderator agent using Gemini 2.5 Pro via OpenRouter."""
    # Use OpenAI provider (OpenRouter is OpenAI-compatible) for Gemini model
    moderator_llm = OpenAICompletion(
        model="google/gemini-2.5-pro",
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        default_headers={
            "HTTP-Referer": "https://github.com/your-repo",
            "X-Title": "CrewAI Debate Example"
        }
    )
    
    return Agent(
        role="Moderator",
        goal="Monitor the debate quality and decide when sufficient discussion has occurred. Then provide a balanced, comprehensive summary.",
        backstory="""You are an impartial moderator with expertise in managing debates and 
        synthesizing complex discussions. Your role is to:
        1. Evaluate after each round whether the debate has reached sufficient depth and resolution
        2. Decide if more rounds are needed or if the debate is complete
        3. Provide a fair, balanced summary that captures the key points, strengths, and 
           weaknesses of each position when the debate concludes.""",
        verbose=True,
        allow_delegation=False,
        llm=moderator_llm
    )


def run_debate_round(topic: str, round_num: int, previous_tasks: list, max_rounds: int, 
                    logger: logging.Logger = None, conversation_log: str = None):
    """Run a single debate round and return tasks and moderator decision.
    
    Args:
        topic: The debate topic
        round_num: Current round number
        previous_tasks: List of previous tasks
        max_rounds: Maximum number of rounds
        logger: Logger instance for logging
        conversation_log: Path to conversation log file
        
    Returns:
        tuple: (tasks_list, moderator_decision, should_continue)
    """
    if logger is None:
        logger = logging.getLogger("debate_logger")
    proponent = create_proponent_agent()
    opponent = create_opponent_agent()
    moderator = create_moderator_agent()
    
    tasks = []
    
    # Get the last proponent statement (if any)
    # Find the last task that was assigned to a proponent agent
    last_proponent_task = None
    if previous_tasks:
        for task in reversed(previous_tasks):
            if hasattr(task, 'agent') and task.agent and hasattr(task.agent, 'role') and task.agent.role == "Proponent":
                last_proponent_task = task
                break
    
    if round_num == 1:
        # Opening: Proponent makes initial argument
        proponent_description = f"""Build a strong, well-reasoned opening argument in favor of the following topic:
            
            Topic: {topic}
            
            This is your opening statement. Your argument should:
            - Be logically structured and coherent
            - Include supporting evidence and reasoning
            - Address potential counterarguments
            - Be persuasive and compelling
            
            Present your complete opening argument clearly and thoroughly."""
        
        logger.info(f"Round {round_num} - Creating Proponent opening task")
        logger.info(f"Task description: {proponent_description[:200]}...")
        
        proponent_task = Task(
            description=proponent_description,
            agent=proponent,
            expected_output="A comprehensive, well-structured opening argument supporting the topic"
        )
        tasks.append(proponent_task)
        last_proponent_task = proponent_task
    else:
        # Proponent responds to opponent's critique
        proponent_description = f"""Round {round_num}: Defend your position and respond to the Opponent's critique about the following topic:
            
            Topic: {topic}
            
            The Opponent has challenged your argument. Now you must:
            - Address their specific criticisms directly
            - Strengthen your position with additional evidence or reasoning
            - Refute their counterarguments
            - Clarify any misunderstandings
            - Reinforce the strongest points of your case
            
            Provide a strong rebuttal that defends and strengthens your position."""
        
        logger.info(f"Round {round_num} - Creating Proponent rebuttal task")
        logger.info(f"Context: {len(previous_tasks)} previous tasks")
        logger.info(f"Task description: {proponent_description[:200]}...")
        
        proponent_task = Task(
            description=proponent_description,
            agent=proponent,
            expected_output=f"A strong rebuttal and defense for round {round_num}",
            context=previous_tasks
        )
        tasks.append(proponent_task)
        last_proponent_task = proponent_task
    
    # Opponent responds
    # Opponent sees all previous tasks to understand the full debate context
    opponent_context = previous_tasks + ([last_proponent_task] if last_proponent_task else [])
    opponent_description = f"""Round {round_num}: Critically analyze and respond to the Proponent's argument about the following topic:
        
        Topic: {topic}
        
        Review the Proponent's most recent argument and the full debate history. Consider:
        - Identify logical fallacies or weak reasoning in their current and previous arguments
        - Point out unsupported claims or assumptions
        - Highlight gaps in evidence or reasoning
        - Note any inconsistencies or contradictions across their statements
        - Present strong counterarguments
        - Expose vulnerabilities in their position
        
        Provide a detailed critique and counterargument. Be specific and direct in your response."""
    
    logger.info(f"Round {round_num} - Creating Opponent critique task")
    logger.info(f"Context: {len(opponent_context)} tasks (previous + current proponent)")
    logger.info(f"Task description: {opponent_description[:200]}...")
    
    opponent_task = Task(
        description=opponent_description,
        agent=opponent,
        expected_output=f"A detailed critique and counterargument for round {round_num}",
        context=opponent_context
    )
    tasks.append(opponent_task)
    
    # Moderator evaluates if more rounds are needed
    all_previous_tasks = previous_tasks + tasks
    evaluation_description = f"""Evaluate the current state of the debate about the following topic:
        
        Topic: {topic}
        
        You have observed {round_num} round(s) of debate. Review the exchange so far and decide:
        
        1. Has the debate reached sufficient depth and resolution?
        2. Are there still important points that need to be addressed?
        3. Would additional rounds add value, or is the discussion becoming repetitive?
        
        IMPORTANT: Respond with ONLY one word: "CONTINUE" if more debate is needed, or "DONE" if 
        the debate has reached sufficient depth. Maximum rounds allowed: {max_rounds}.
        
        If this is round {max_rounds}, you must respond with "DONE"."""
    
    logger.info(f"Round {round_num} - Creating Moderator evaluation task")
    logger.info(f"Context: {len(all_previous_tasks)} total tasks")
    logger.info(f"Task description: {evaluation_description[:200]}...")
    
    evaluation_task = Task(
        description=evaluation_description,
        agent=moderator,
        expected_output="Either 'CONTINUE' or 'DONE' - one word only",
        context=all_previous_tasks
    )
    tasks.append(evaluation_task)
    
    # Create crew for this round
    logger.info(f"Round {round_num} - Creating crew with {len(tasks)} tasks")
    logger.info(f"Agents: Proponent, Opponent, Moderator")
    
    crew = Crew(
        agents=[proponent, opponent, moderator],
        tasks=tasks,
        process=Process.sequential,
        verbose=True
    )
    
    logger.info(f"Round {round_num} - Executing crew...")
    result = crew.kickoff()
    logger.info(f"Round {round_num} - Crew execution completed")
    logger.info(f"Result type: {type(result)}")
    logger.info(f"Result preview: {str(result)[:500]}...")
    
    # Write conversation entries for each completed task
    # Outputs are in result.tasks_output, not on task objects
    if conversation_log and hasattr(result, 'tasks_output'):
        for i, task_output in enumerate(result.tasks_output):
            # Match task output to the original task by index
            if i < len(tasks):
                task = tasks[i]
                agent_role = task_output.agent if hasattr(task_output, 'agent') else (task.agent.role if hasattr(task, 'agent') and task.agent else 'Unknown')
                input_text = task.description
                
                # Get context if available (check if it's actually a list/iterable)
                if hasattr(task, 'context') and task.context:
                    try:
                        # Check if context is iterable and not a special CrewAI object
                        context_list = list(task.context) if not isinstance(task.context, str) else []
                        if context_list:
                            context_summary = []
                            # Try to get context from previous task outputs
                            for ctx_task in context_list:
                                # Look for matching task output in previous rounds
                                # For now, just note that context was provided
                                if hasattr(ctx_task, 'agent') and ctx_task.agent:
                                    ctx_role = ctx_task.agent.role
                                    context_summary.append(f"Context includes previous {ctx_role} statement")
                            
                            if context_summary:
                                input_text += "\n\n**Context:**\n" + "\n".join(f"- {s}" for s in context_summary)
                    except (TypeError, AttributeError):
                        # Context is not iterable (e.g., _NotSpecified), skip it
                        pass
                
                # Get output from TaskOutput object
                output_text = task_output.raw if hasattr(task_output, 'raw') and task_output.raw else str(task_output)
                is_eval = (agent_role == "Moderator" and ("CONTINUE" in output_text.upper() or "DONE" in output_text.upper()))
                
                logger.info(f"Writing conversation entry for {agent_role} - output length: {len(output_text)}")
                
                write_conversation_entry(
                    conversation_log, 
                    round_num, 
                    agent_role, 
                    input_text, 
                    output_text,
                    is_evaluation=is_eval
                )
    
    # Extract moderator's decision from the result
    decision_text = str(result).strip().upper()
    should_continue = "CONTINUE" in decision_text and round_num < max_rounds
    
    logger.info(f"Round {round_num} - Decision extracted: {decision_text}")
    logger.info(f"Round {round_num} - Should continue: {should_continue}")
    
    return tasks, should_continue, result


def create_final_summary_crew(topic: str, all_debate_tasks: list):
    """Create a crew to generate the final summary."""
    moderator = create_moderator_agent()
    
    summary_task = Task(
        description=f"""Synthesize the complete debate about the following topic:
        
        Topic: {topic}
        
        Review the entire debate exchange and write a balanced, comprehensive summary that:
        - Fairly represents both perspectives across all rounds
        - Highlights the key points and arguments from each side
        - Identifies the strengths and weaknesses of each position
        - Notes how the debate evolved through the rounds
        - Provides an objective assessment of the overall discussion
        - Offers insights into which side made stronger points
        
        Your summary should be clear, balanced, and useful for understanding the full debate.""",
        agent=moderator,
        expected_output="A balanced, comprehensive summary of the entire debate that fairly represents both perspectives",
        context=all_debate_tasks
    )
    
    crew = Crew(
        agents=[moderator],
        tasks=[summary_task],
        process=Process.sequential,
        verbose=True
    )
    
    return crew


def main(topic: str = None):
    """Main function to run the debate crew.
    
    Args:
        topic: The debate topic (optional, will prompt if not provided)
    """
    print("=" * 60)
    print("CrewAI Debate Example: Role-Based Collaboration")
    print("=" * 60)
    print()
    print(f"Maximum debate rounds (from .env): {MAX_ROUNDS}")
    print("The Moderator will decide when the debate is complete.")
    print()
    
    # Prompt for topic if not provided
    if not topic:
        topic = input("Enter the debate topic: ").strip()
        if not topic:
            topic = "Should artificial intelligence be regulated by governments?"
            print(f"\nNo topic provided, using default: {topic}\n")
        else:
            print()
    else:
        print(f"Debate topic: {topic}\n")
    
    # Setup logging
    logger, log_filename, conversation_log_filename = setup_logging(topic)
    
    logger.info("=" * 80)
    logger.info("DEBATE SESSION STARTED")
    logger.info("=" * 80)
    logger.info(f"Topic: {topic}")
    logger.info(f"Maximum Rounds: {MAX_ROUNDS}")
    logger.info(f"OpenRouter Base URL: {OPENROUTER_BASE_URL}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    
    print(f"\nStarting debate on: {topic}\n")
    print(f"Technical log: {log_filename}")
    print(f"Conversation log: {conversation_log_filename}\n")
    print("-" * 60)
    print()
    
    # Run debate rounds until moderator says done or max rounds reached
    all_debate_tasks = []
    round_num = 1
    should_continue = True
    
    while should_continue and round_num <= MAX_ROUNDS:
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"ROUND {round_num} of {MAX_ROUNDS} (maximum)")
        logger.info("=" * 80)
        logger.info("")
        
        print(f"\n{'='*60}")
        print(f"ROUND {round_num} of {MAX_ROUNDS} (maximum)")
        print(f"{'='*60}\n")
        
        logger.info(f"Starting Round {round_num}")
        logger.info(f"Previous tasks count: {len(all_debate_tasks)}")
        
        tasks, should_continue, round_result = run_debate_round(
            topic, round_num, all_debate_tasks, MAX_ROUNDS, logger, conversation_log_filename
        )
        
        all_debate_tasks.extend(tasks[:-1])  # Add all tasks except the evaluation
        
        # Log round results
        logger.info("")
        logger.info("-" * 80)
        logger.info(f"ROUND {round_num} RESULTS")
        logger.info("-" * 80)
        for i, task in enumerate(tasks[:-1], 1):  # Exclude evaluation task
            task_output = getattr(task, 'output', 'Not yet executed')
            agent_role = task.agent.role if hasattr(task, 'agent') and task.agent else 'Unknown'
            logger.info(f"Task {i} ({agent_role}): {str(task_output)[:200]}...")
        logger.info("")
        
        # Check moderator's decision
        decision_text = str(round_result).strip().upper()
        logger.info(f"Moderator evaluation result: {round_result}")
        logger.info(f"Decision extracted: {decision_text}")
        
        if "DONE" in decision_text or round_num >= MAX_ROUNDS:
            logger.info("")
            logger.info("=" * 80)
            logger.info("DEBATE COMPLETE - Moderator decided to end debate")
            logger.info(f"Total rounds completed: {round_num}")
            logger.info("=" * 80)
            logger.info("")
            
            print(f"\n{'='*60}")
            print(f"Moderator has decided: DEBATE COMPLETE")
            print(f"Total rounds: {round_num}")
            print(f"{'='*60}\n")
            should_continue = False
        else:
            logger.info("")
            logger.info("=" * 80)
            logger.info("DEBATE CONTINUING - Moderator decided more rounds needed")
            logger.info("=" * 80)
            logger.info("")
            
            print(f"\n{'='*60}")
            print(f"Moderator has decided: CONTINUE")
            print(f"{'='*60}\n")
            round_num += 1
    
    # Generate final summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("GENERATING FINAL SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total debate tasks: {len(all_debate_tasks)}")
    logger.info("")
    
    print(f"\n{'='*60}")
    print("GENERATING FINAL SUMMARY")
    print(f"{'='*60}\n")
    
    summary_crew = create_final_summary_crew(topic, all_debate_tasks)
    final_summary = summary_crew.kickoff()
    
    # Write final summary to conversation log
    if conversation_log_filename:
        with open(conversation_log_filename, 'a', encoding='utf-8') as f:
            f.write("# Final Summary\n\n")
            f.write(f"{str(final_summary)}\n\n")
            f.write("---\n\n")
            f.write(f"**Ended:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Total Rounds:** {round_num}\n\n")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("FINAL DEBATE SUMMARY")
    logger.info("=" * 80)
    logger.info("")
    logger.info(str(final_summary))
    logger.info("")
    logger.info("=" * 80)
    logger.info("DEBATE SESSION ENDED")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {log_filename}")
    logger.info(f"Conversation log: {conversation_log_filename}")
    logger.info("=" * 80)
    
    print("\n" + "=" * 60)
    print("DEBATE SUMMARY")
    print("=" * 60)
    print()
    print(final_summary)
    print()
    print(f"Technical log: {log_filename}")
    print(f"Conversation log: {conversation_log_filename}")
    print()


if __name__ == "__main__":
    import sys
    # Get topic from command line argument or use default
    topic = sys.argv[1] if len(sys.argv) > 1 else None
    main(topic=topic)

