import re

def extract_answer(text):
    # Convert to lowercase for matching
    marker = "the final answer is"
    text_lower = text.lower()
    start_index = text_lower.find(marker)
    
    if start_index != -1:
        # Extract from the original text, preserving the original case
        answer_start = start_index + len(marker)
        # Extract until a period or newline character
        answer = text[answer_start:].strip()
        # If there is a period in the answer, only take the content before the period
        period_index = answer.find('.')
        if period_index != -1:
            answer = answer[:period_index]
        return answer.strip()
    else:
        return ""

def clean_relations(string, entity_id, head_relations):
    pattern = r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}"
    relations=[]
    for match in re.finditer(pattern, string):
        relation = match.group("relation").strip()
        if ';' in relation:
            continue
        score = match.group("score")
        if not relation or not score:
            return False, "output uncompleted.."
        try:
            score = float(score)
        except ValueError:
            return False, "Invalid score"
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": True})
        else:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": False})
    if not relations:
        return False, "No relations found"
    return True, relations

def construct_relation_prune_prompt(question, entity_name, total_relations, width = 3):
    return extract_relation_prompt % (width, width) + question + '\nTopic Entity: ' + entity_name + '\nRelations: '+ '; '.join(total_relations) + "\nA: "

def extract_entity_names(output):
    cleaned_output = output.strip()
    entity_names = [name.strip() for name in cleaned_output.split(',')]
    # Filter out empty strings and 'Unknown'
    entity_names = [name for name in entity_names if name]
    return entity_names