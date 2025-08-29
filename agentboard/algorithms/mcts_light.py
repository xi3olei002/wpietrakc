import pdb

from common.registry import registry
# from rouge import Rouge
import json
import random
import re
import logging
import argparse
import numpy as np

logging.basicConfig(filename='mcts_50it.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

@registry.register_algorithm("MCTS_Light")
class MCTS_Light:  # the agent should receive goal, state and action, then return the next state
    def __init__(self,
                 llm_model,
                 prompt_path
                 ):
        
        self.llm_model = llm_model
        self.world_model = llm_model
        self.prompts = json.load(open(prompt_path, 'r'))
        self.system_message = self.prompts["system_msg"]
        
        self.task = "dp" if "dp" in prompt_path else "pf"
        
        self.problem_size = 6
    
    def make_prompt(self, node):
        query = ""
        query += self.prompts["instruction"] + '\n'
        query += "\nHere is an example:\n" + self.prompts["examples"][0] + '\n'
        
        input_prompt = query + 'Question:' + node.question
        
        trajectory = []
        while node:
            if node.state['action']:
                trajectory.append(node.state['action'])
                
            node = node.parent
        
        trajectory = [action for action in reversed(trajectory)]
        trajectory = list(trajectory)
        trajectory += [None] * (self.problem_size - len(trajectory))
        input_prompt = input_prompt + f"\nYou could only change None in the constraints, constraints:{trajectory}\nAnswer="
        return input_prompt

    def parse_integer_lists(self, string):
        pattern = r'\[([\d\s,]+)\]'  # Regular expression pattern to match integer lists
        matches = re.findall(pattern, string)  # Find all matches

        # Parse each match
        result = []
        for match in matches:
            integers = re.findall(r'\d+', match)  # Find all integers in the match
            integers = [int(num) for num in integers]  # Convert strings to integers
            result.append(integers)

        return result
    def run(self, question):
        
        args = {
            "n_generate_sample": 20,
            "depth_limit": self.problem_size,
            "iterations": 20
        }
        
        #dict to object
        
        self.prompts["question"] = question
        
        args = argparse.Namespace(**args)
        all_terminals = self.mcts_search(args, question, args.iterations, args.depth_limit)
        
        
        if len(all_terminals) > 0:
            
            sorted_terminals = sorted(all_terminals, key=lambda x: x.reward, reverse=True)
            
            best_node = sorted_terminals[0]
            
            answer_prompt = self.make_prompt(best_node) + "\n output="
            success, answer = self.llm_model.generate(self.system_message, answer_prompt)
            
            if success:
                result_lists = self.parse_integer_lists(answer)
                if len(result_lists) > 0:
                    return True, result_lists[-1]
        
        return False, None

    def expand_node(self, node, args, depth_limit):
        if node.depth >= depth_limit:
            logging.info("Depth limit reached")
            print("Depth limit reached")
            node.is_terminal = True
            return
        new_nodes = self.generate_new_states(node, args)
        node.children.extend(new_nodes)
        
        
    def get_samples(self, n_generate_sample, prompt, stop=None):
        if stop is not None:
            config = {"n": n_generate_sample, "stop": stop, "max_tokens": 30, "temperature": 0.7}
        else:
            config = {"n": n_generate_sample, "max_tokens": 30, "temperature": 0.7}
            
        success, samples = self.llm_model.generate_with_config(self.prompts["system_msg"], prompt, config)
        return success, samples
    
    def eval(self, trajectory):
        if self.task == "dp":
            return self.eval_dp(trajectory)
        elif self.task == "pf":
            return self.eval_pf(trajectory)
        else:
            raise NotImplementedError
    
    def find_numbers(self, text):
        # Regular expression pattern to match integers and floats
        # Explanation:
        # -?        Optional negative sign
        # \d+       One or more digits
        # (?:       Start of non-capturing group for decimal part
        #   \.\d+   Decimal point followed by one or more digits
        # )?        The decimal part is optional
        pattern = r'-?\d+(?:\.\d+)?'
        
        # Find all matches using re.findall
        numbers = re.findall(pattern, text)
        
        # Convert all found strings to float or int as appropriate
        converted_numbers = []
        for number in numbers:
            # Convert to float if it contains a decimal point, else to int
            if '.' in number:
                converted_numbers.append(float(number))
            else:
                converted_numbers.append(int(number))
        
        return converted_numbers
    
    def eval_dp(self, action_list):
        input = eval(self.prompts["question"].split("=")[1])
        
        num_sum = sum([input[i] for i in range(len(action_list)) if action_list[i] == 1])
        
        for i in range(len(action_list)-1):
            if action_list[i] == 1 and action_list[i+1] == 1:
                return True, -50
        return True, num_sum
        
        prompt = self.make_prompt(self.prompts)
        
            # if "Observation" in item and item["Observation"] is not None:
            #     prompt += f"\nObservation: {item['Observation']}"
        
        prompt += f"\n{self.prompts['evaluation_prompt']}\nAnswer: {action_list}. "
        prompt += f"\nReward:"
        
        config = {"stop": '', "max_tokens": 50}
        success, output = self.llm_model.generate_with_config(self.prompts["system_msg"], prompt, config)
        
        if "is" in output:
            output = "is".join(output.split("is")[1:])
        if ":" in output:
            output = ":".join(output.split(":")[1:])
        numbers = self.find_numbers(output)
        
        if len(numbers) > 0:
            return True, numbers[0]
        else:
            return False, None
        
    def eval_pf(self, trajectory):
        return True, 0
    
    def verify_action(self, action_line, node):
        depth = node.depth + 1
        lists = self.parse_integer_lists(action_line)
        
        if len(lists) == 0:
            return False, None
        else:
            action_line = lists[0]
        
        trajectory = []
        while node:
            if node.state['action']:
                trajectory.append(node.state['action'])
                
            node = node.parent
        
        trajectory = [action for action in reversed(trajectory)]
        
        if len(action_line) < depth:
            return False, None
        else:
            if action_line[:depth-1] != trajectory: 
                return False, None
            action = action_line[depth-1]
            return True, action

    def generate_new_states(self, node, args):
        prompt = self.make_prompt(node) 
        success, sampled_actions = self.get_samples(args.n_generate_sample, prompt, stop=",]")
        logging.info(f"SAMPLED ACTION: {sampled_actions}")
        
        unique_states = {}  # Store unique states here
        
        sampled_actions = list(set(sampled_actions))
        
        verified_actions = []
        for action_line in sampled_actions:
            verified, action = self.verify_action(action_line, node)
            if verified:
                verified_actions.append(action)
        
        verified_actions = list(set(verified_actions))
        for action in verified_actions:
            new_state = node.state.copy()  # Make a copy of the parent node's state

            # reformat action here
            action = int(action)
            
            original_state = new_state['observation']

            # Use thought and action to form a unique key
            unique_key = action
            
            if unique_key in unique_states:
                continue  # Skip if this state already exists
            
            if action:
                obs = original_state + [action]
                done = len(obs) == self.problem_size
                succes, r = self.eval(obs)

                # Update the new state dictionary
                new_state['action'] = action
                new_state['observation'] = obs

                new_node = Node(state=new_state, question=node.question, parent=node)
                new_node.is_terminal = done
                new_node.reward = r
                
                unique_states[unique_key] = new_node  # Add this state to unique_states
                logging.info(f"NEW NODE: {new_node}")
        return list(unique_states.values())  # Return unique nodes as a list
    
    def collect_all_nodes(self, node):
        """Recursively collect all nodes starting from the given node."""
        nodes = [node]
        for child in node.children:
            nodes.extend(self.collect_all_nodes(child))
        return nodes

    
    def select_node(self, node):
        while node and node.children:
            logging.info(f"Selecting from {len(node.children)} children at depth {node.depth}.")
        
            terminal_children = [child for child in node.children if child.is_terminal]
            terminal_status = [child.is_terminal for child in node.children]
            
            if len(terminal_children) == len(node.children):
                logging.info(f"All children are terminal at depth {node.depth}. Backtracking...")
                if node.parent:  
                    node.parent.children.remove(node)
                node = node.parent  
                continue  
            
            node = max((child for child in node.children if not child.is_terminal), key=lambda child: child.uct(), default=None)
            
            while node.is_terminal:
                node = max((child for child in node.parent.children if not child.is_terminal), key=lambda child: child.uct(), default=None)
            
            logging.info(f"Selected node at depth {node.depth} with UCT {node.uct()}.")
        
        return node  # This will return None if all paths from the root are exhausted
            
    def backpropagate(self, node, value):
        while node:
            node.visits += 1
            node.value = (node.value * (node.visits - 1) + value) / node.visits
            logging.info(f"Backpropagating at depth {node.depth}. New value: {node.value}.")

            node = node.parent
            
    def evaluate_node(self, node):
        values = []
        
        for child in node.children:
            values.append(child.reward)
        
        logging.info(f"Length of votes: {len(values)}")
        logging.info(f"Length of node.children: {len(node.children)}")
        
        if len(values) == 0:
            return node.reward
        
        value = sum(values) / len(values)
        
        return value
        
    def simulate(self, path, args, depth_limit):
        node = path[-1]
        logging.info(f"Simulating from node {node}...")
        while True:
            if node is not None and len(node.children) == 0:
                self.expand_node(node, args, depth_limit)
            if node.is_terminal or len(node.children) == 0:
                return
            rewards = np.array([child.reward for child in node.children])
            # choose the child with the highest reward
            node = node.children[np.argmax(rewards)]
            logging.info(f"Simulation selected {node}")
            path.append(node)
        return
    
    def mcts_search(self, args, question, iterations, depth_limit):
        root = Node(state=None, question=question)
        all_nodes = []
        failed_trajectories = []
        terminals = []
        
        for i in range(iterations):
            logging.info(f"Iteration {i + 1}...")
            node = self.select_node(root)
            
            while node is None or (node.is_terminal):
                logging.info(f"Need to backtrack or terminal node with reward 0 found at iteration {i + 1}, reselecting...")
                node = self.select_node(root)
                
            if node is None:
                logging.info("All paths lead to terminal nodes with reward 0. Ending search.")
                break
            
            if len(node.children) == 0:
                self.expand_node(node, args, depth_limit)
            
            
            while node.is_terminal:
                logging.info(f"Depth limit node found at iteration {i + 1}, reselecting...")
                node = self.select_node(root)
                if len(node.children) == 0:
                    self.expand_node(node, args, depth_limit)
            
            path = [node]
            self.simulate(path, args, depth_limit)
            
            value = path[-1].reward
            
            self.backpropagate(node, value)
            
            terminals = []
            all_nodes = [(node, node.value) for node in self.collect_all_nodes(root)]
            
            for j, (node, value) in enumerate(all_nodes):
                if node.is_terminal and node not in terminals:
                    terminals.append(node)

            for j, (node, value) in enumerate(all_nodes):
                logging.info(f"Node {j+1}: {str(node)}")

            logging.info(f"State of all_nodes after iteration {i + 1}: {all_nodes}")
        
        
        return terminals
    
    
    @classmethod
    def from_config(cls, llm_model, config):
        return cls(llm_model, config["prompt_path"])
    
class Node:
    def __init__(self, state, question, parent=None):
        self.state = {'action': None, 'observation': []} if state is None else state
        self.parent = parent
        self.question = question
        self.children = []
        self.visits = 0
        self.value = 0
        self.depth = 0 if parent is None else parent.depth + 1
        self.is_terminal = False
        self.reward = 0
        self.exhausted = False # If all children are terminal

    def uct(self):
        if self.visits == 0:
            #return float('inf')
            return self.value * 2
        return self.value / self.visits + np.sqrt(2 * np.log(self.parent.visits) / self.visits)
    
    def uct_with_depth(self, C1=1, C2=1):
        if self.visits == 0:
            return self.value
        exploitation_term = self.value / self.visits
        exploration_term = np.sqrt(2 * np.log(self.parent.visits) / self.visits)
        depth_term = self.depth
        return exploitation_term + C1 * exploration_term + C2 * depth_term

    def __str__(self):
        return f"Node(depth={self.depth}, reward={self.reward:.2f}, value={self.value}, is_terminal={self.is_terminal}, action={self.state['action']}, observation={str(self.state['observation'])}"
    
    def to_dict(self):
        return {
            'state': self.state,
            'question': self.question,
            'parent': self.parent.to_dict() if self.parent else None,
            'children': [child.to_dict() for child in self.children],
            'visits': self.visits,
            'value': self.value,
            'depth': self.depth,
            'is_terminal': self.is_terminal,
            'reward': self.reward,
        }
 