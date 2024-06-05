from Pipelines.Trainer import *
from Pipelines.Scorer import *

if __name__=='__main__':
    trainer = Trainer('base_train.csv')
    trainer.orchestrator()
    scorer=Scorer('base_val.csv')
    scorer.orchestrator()