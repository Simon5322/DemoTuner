

from transformers import pipeline

question = "Should share_buffers to larger on a specific condition?"
context = " Larger settings for shared_buffers "# usually require a corresponding increase in max_wal_size"

question = "How many programming languages does BLOOM support?"
context = "BLOOM has 176 billion parameters and can generate text in 46 languages natural languages and 13 programming languages."
#
qa_pipeline = pipeline(
   'question-answering',
    model='deepset/roberta-base-squad2',
   tokenizer='deepset/roberta-base-squad2')
#
#
print(qa_pipeline(question=question, context=context))
