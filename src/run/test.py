from transformers import pipeline

config = "shared_buffers"
zsc_pipeline= pipeline(
    'zero-shot-classification',
    model="facebook/bart-large-mnli")
qa_pipeline = pipeline(
    'question-answering',
    model='deepset/roberta-base-squad2',
    tokenizer='deepset/roberta-base-squad2')
# print(qa_pipeline(question="Is shared_buffer affect others or be affected",
text1 = "a reasonable starting value for shared_buffers is 25% of the memory in your system,if RAM is 1GB"
text2 = "Larger settings for shared_buffers usually require a corresponding increase in max_wal_size"
text3 = "Sets the maximum number of temporary buffers used by each database session"
text4 = "Note that when autovacuum runs, up to autovacuum_max_workers times this memory may be allocated, so be careful not to set the default value too high"
label = ["yes", "no"]
lable2 = ["index", "sort", "configuration", "workload", "other"]
label3 = [f"{config} affects other configs", f"{config} is affected by other condition", "not affected "]

affected = ""
affecting = ""
condition = [affected, affecting]

print(zsc_pipeline(text2, label3))
# score = affectType["scores"]
# max_index = score.index(max(score))
# max_lable = affectType["labels"][max_index]
# print(max_lable)
# q_result = qa(question=f"{config} affect or be affected by what", context=text1)
# q_answer = q_result["answer"]
# q_score = q_result["score"]
#
# if max_lable == label3[0]:
#     affecting = q_answer
#     condition[1] = affecting
#
# elif max_lable == label3[1]:
#     affected = q_answer
#     condition[0] = affected
# else:
#     print("not valid lable")
#
# print(condition[0], condition[1])


def get_passage_affectType(config, exp_passage):
    label3 = [f"{config} affects other configs", f"{config} is affected by other condition", "not affected "]
    affected = ""
    affecting = ""
    condition = [affecting, affected]

    affectType = zsc_pipeline(exp_passage, label3)
    # print(affectType)
    score = affectType["scores"]
    max_index = score.index(max(score))  # always 0
    max_label = affectType["labels"][max_index]
    # print(max_label)
    q_result = qa_pipeline(question=f"{config} affect or be affected by what", context=exp_passage)
    # print(q_result)
    q_answer = q_result["answer"]
    # print(q_answer)
    # q_score = q_result["score"]
    index = -1
    if max_label == label3[0]:
        index = 0
        affecting = q_answer
        condition[index] = affecting

    elif max_label == label3[1]:
        index = 1
        affected = q_answer
        condition[index] = affected
    else:
        print("not valid lable")

    return {max_label, condition[index]}


print(get_passage_affectType(config, text2))