from abc import ABC

# from run import conditionType


class ABSCondition(ABC):
    def __init__(self, Type):
        self.conditionType = Type

    def if_meet_condition(self):
        pass


class Condition(ABSCondition):
    def __init__(self, Type):
        super().__init__(Type)
        self.conditionType = Type

    def has_meet_condition(self):
        pass

