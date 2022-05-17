from collections import OrderedDict
import re
import sys, io, csv

# new_dict_comp = {n:0 for n in list(range(gym_env.action_space.n))}
# new_dict_comp[policy_outputs["action"].item()] += 1

# Used to log actions
class ActionLog:
    def __init__(self, env):
        self.env = env
        self.action_dict_comp = {n:0 for n in list(range(env.action_space.n))}
        self.action_names_dict_comp = {}
        self.action_meaning_dict = {}
        self.num_actions = 0
        super().__init__()

        
    # take an action and adds it to a dictionary of actions
    def record_action(self, action):
        self.num_actions += 1

        # update actions dictionary
        if action in self.action_dict_comp.keys():
            self.action_dict_comp[action] += 1
        else:
            self.action_dict_comp[action] = 1

    
    # writes a dictionary to a csv file
    def write_to_csv(self, dictionary):
        # open file in write mode
        f = open('agentActions.csv', 'w')
        writer = csv.writer(f)
        # save lines of the dictionary into the csv
        for key in dictionary:
            writer.writerow([key, dictionary[key]])


    # remaps the action dictionary from having the action keys as numbers
    # and creates a new array with the keys set as the action names
    def remap_dictionary(self):
        # save stdout because we will be modifying the system IO
        old_stdout = sys.stdout

        # save string IO
        new_stdout = io.StringIO()

        # modify system print to be the saved 
        sys.stdout = new_stdout

        # grab the print value from the function
        action_meanings = self.env.print_action_meanings()

        # grab text value from the print
        output = new_stdout.getvalue()

        # put the normal stdout back
        sys.stdout = old_stdout

        # split output to parse values from the print
        output = output.split(" ")

        # this parses the newlines to save as a new list
        outputLength = len(output)
        for i in range(1, outputLength):
            if i == 1:
                key = output[i-1]
            else:
                key = output[i-1].splitlines()[1]
            x = output[i].splitlines()
            val = x[0]
            
            self.action_meaning_dict[key] = val

        # save parsed disctionary into a new dictionary
        for key in self.action_dict_comp:
            nameKey = self.action_meaning_dict[str(key)]
            self.action_names_dict_comp[nameKey] = self.action_dict_comp[key]


    # display actor actions and counts
    def print_actions(self):
        print("\nAgent Actions (action:count): ")

        self.remap_dictionary()

        self.write_to_csv(self.action_names_dict_comp)

        for key in self.action_dict_comp:
            print(key, ":", self.action_dict_comp[key], ", ", end='', sep='')

        print("\n\nNumber of agent actions: ", self.num_actions, "\n")

    
    # reset member variables
    def clear_actions(self):
        self.action_dict_comp = {}
        self.action_names_dict_comp = {}
        self.action_meaning_dict = {}
        self.num_actions = 0
