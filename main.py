from resources.agent import Agent
from resources.input_filter import InputFilter, Action
from resources.output_filter import OutputFilter
from utils.generate_response import generate_response

def train(episodes=5):
    agent = Agent()
    output_filter = OutputFilter()

    for episode in range(1, episodes + 1):
        input_filter = InputFilter(
            prompt="What is sun made up of?"
        )

        print(f"\n=== Episode {episode}/{episodes} ===")
        print(f"\nPrompt:\n{input_filter.prompt}\n")

        state_old = agent.get_state(input_filter, action=None)

        final_move = agent.get_action(state_old)
        print(f"Chosen action: {Action(final_move.index(1)).name}")

        state_new = agent.get_state(input_filter, Action(final_move.index(1)))
        prompt = input_filter.prompt
        flags = input_filter.flags

        if prompt is None:
            response = "I'm sorry, I'm not supposed to answer such questions."
        else:
            response = generate_response(prompt)
        
        reward = agent.get_reward(output_filter, response, flags)
        print(f"Generated response:\n{response}")
        print(f"Reward: {reward}")

        agent.train_short_memory(state_old, final_move, reward, state_new)
        agent.remember(state_old, final_move, reward, state_new)

        print(f"Episode {episode} completed.\n")

    print("Training finished.")


if __name__ == '__main__':
    train()