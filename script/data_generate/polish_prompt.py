def polish_prompt(text,type='abstract'):
    if type == 'abstract':
        prompt_list = [
        'Can you help me revise the abstract? Please response directly with the revised abstract: {text}',
        'Please revise the abstract, and response directly with the revised abstract: {text}',
        'Can you check if the flow of the abstract makes sense? Please response directly with the revised abstract: {text}',
        'Please revise the abstract to make it more logical, response it directly with the revised abstract: {text}',
        'Please revise the abstract to make it more formal and academic, response it directly with the revised abstract: {text}'
        ]
    else:
        prompt_list = [
        'Can you help me revise the meta review? Please response directly with the revised meta review: {text}',
        'Please revise the meta review, and response directly with the revised meta review: {text}',
        'Can you check if the flow of the meta review makes sense? Please response directly with the revised meta review: {text}',
        'Please revise the meta review to make it more logical, response it directly with the revised abstract: {text}',
        'Please revise the meta review to make it more formal and academic, response it directly with the revised meta review: {text}'
        ]
    return [prompt_list[i].format(text=text) for i in range(len(prompt_list))]
