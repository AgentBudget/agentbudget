from agentbudget import AgentBudget
class FakeUnrecognizedResponse:

    def init(self):
        self.weird_usage_field = {"in": 50000, "out": 50000} # not "usage"
        self.weird_model_field = "some-expensive-model-v9" # not "model"

budget = AgentBudget(max_spend="$1.00")
session = budget.session()

for i in range(5):
    session.wrap(FakeUnrecognizedResponse())

print(session.report())