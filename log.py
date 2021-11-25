# Used to log actions
class ActionLog:
    def __init__(self):
        self.action_count = {}
        self.num_actions = 0
        super().__init__()

        
    # take an action and adds it to a dictionary of actions
    def record_action(self, action):
        self.num_actions += 1

        # update actions dictionary
        if action in self.action_count.keys():
            self.action_count[action] += 1
        else:
            self.action_count[action] = 1

    
    # display actor actions and counts
    def print_actions(self):
        print("\nAgent Actions (action:count): ", end='')

        for key in self.action_count:
            print(key, ":", self.action_count[key], ", ", end='', sep='')

        print("\n\nNumber of agent actions: ", self.num_actions, "\n")

    
    # reset member variables
    def clear_actions(self):
        self.action_count = {}
        self.num_actions = 0
