class BasicAgent:
    def __init__(self, n1: int):
        '''
        Simple Agent that, given a state,
         it chops the network at a fixed point.
        :param n1:
        '''
        self.n1 = n1

    def __repr__(self):
        return "BasicAgent"

    def exploit(self, state=None)->int:
        return self.n1