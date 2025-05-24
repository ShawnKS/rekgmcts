import math
import random
import re
from collections import defaultdict

class MCTSNode:
    def __init__(self, entities_info, triples=None, pre_relations=None, pre_head=None, parent=None):
        """
        Initialize MCTS Node
        
        Args:
            entities_info: List[Dict] List of entity information for the current node
                [{
                    'entity_id': str,
                    'entity_name': str
                }, ...]
            triples: List[Dict] Triples on the path
            pre_relations: List[str] Historical relations for retrieving the current entity
            pre_head: int Head marker corresponding to the relation
            parent: MCTSNode Parent node
        """
        # Basic information
        self.entities_info = entities_info
        self.parent = parent
        self.children = []
        
        # Path information
        self.y = triples or []  # All triples on the current path
        self.v = 0.0  # Value score of the current node
        
        # Historical information - only store the relation information used in the last expansion
        self.pre_relations = pre_relations or []
        self.pre_head = pre_head if pre_head is not None else -1
        
        # MCTS statistics
        self.visits = 1
        self.total_reward = 0
        
        # Expansion state
        self.is_fully_expanded = False
        self.unexpanded_entities = list(range(len(entities_info)))  # List of indices of unexpanded entities
        
        # Cache
        self.cached_relations = {}  # entity_id -> relations
        self.expanded_relations = {entity['entity_id']: set() for entity in entities_info}  # Record expanded relations for each entity
        
    def add_child(self, entity_idx, new_entity_info, new_triple, relation, head):
        """
        Add a child node
        
        Args:
            entity_idx: int Index of the entity being expanded
            new_entity_info: Dict Information of the new entity
            new_triple: Dict New triple
            relation: str Relation used
            head: int Direction of the relation
        
        Returns:
            MCTSNode: Newly created child node
        """
        # Create a new list of entity information, removing the entity used for expansion
        new_entities = self.entities_info.copy()
        new_entities.pop(entity_idx)  # Remove the expanded entity
        new_entities.append(new_entity_info)  # Add the new entity
        
        # Create a new list of triples
        new_triples = self.y + [new_triple]
        
        # Create child node
        child = MCTSNode(
            entities_info=new_entities,
            triples=new_triples,
            pre_relations=[relation],  # Only store the relation used in this expansion
            pre_head=head,
            parent=self
        )
        
        # Copy cached_relations from parent node
        child.cached_relations = self.cached_relations.copy()
        
        # Update expanded_relations, only include entities in the new entity list
        child.expanded_relations = {
            entity['entity_id']: set() 
            for entity in new_entities
        }
        
        self.children.append(child)
        # Mark the used relation as expanded
        self.expanded_relations[self.entities_info[entity_idx]['entity_id']].add(relation)
        
        return child
    
    def cache_relations(self, entity_id, relations):
        """Cache the search results of entity relations"""
        self.cached_relations[entity_id] = relations
    
    def get_cached_relations(self, entity_id):
        """Get cached search results of relations"""
        return self.cached_relations.get(entity_id, [])
    
    def mark_entity_fully_expanded(self, entity_idx):
        """Mark an entity as fully expanded"""
        if entity_idx in self.unexpanded_entities:
            self.unexpanded_entities.remove(entity_idx)
            # If all entities are fully expanded, mark the node as fully expanded
            if not self.unexpanded_entities:
                self.is_fully_expanded = True
    
    def get_unexpanded_entity(self):
        """Get information of an unexpanded entity
        
        Returns:
            Tuple[int, Dict]: (Entity index, Entity information)
        """
        if self.unexpanded_entities:
            idx = self.unexpanded_entities[0]
            return idx, self.entities_info[idx]
        return None, None
    
    def get_uct_value(self, exploration_constant):
        """Calculate the UCT value of the node"""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.total_reward / self.visits
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration
    
    def get_exploitation_value(self):
        """Get the exploitation value of the node"""
        return self.total_reward / self.visits if self.visits > 0 else 0
    
    def is_terminal(self, max_depth):
        """Check if it is a terminal node"""
        return (len(self.y) >= max_depth or 
                self.is_fully_expanded and not self.children)
    
    def get_path(self):
        """Get the complete path from the root node to the current node"""
        return self.y
    
    def __repr__(self):
        """String representation of the node"""
        return (f"MCTSNode("
                f"entities={len(self.entities_info)}, "
                f"triples={len(self.y)}, "
                f"value={self.v:.3f}, "
                f"visits={self.visits}, "
                f"reward={self.total_reward:.3f}, "
                f"children={len(self.children)}, "
                f"expanded={self.is_fully_expanded})")

def _construct_triple(source_entity, relation, target_entity, is_head):
    """
    Construct a knowledge graph triple
    
    Args:
        source_entity: Dict Source entity information 
            {'entity_id': str, 'entity_name': str}
        relation: str Relation
        target_entity: Dict Target entity information
            {'entity_id': str, 'entity_name': str}
        is_head: bool Direction of the relation (True indicates target is the head entity)
    
    Returns:
        Tuple: Triple in the format (head_name, relation, tail_name)
    """
    if is_head:
        # Source entity as head entity
        return (source_entity, relation, target_entity)
    else:
        # Target entity as head entity
        return (target_entity, relation, source_entity)

class MCTSPathFinder:
    def __init__(self, question, topic_entities, llm, num_retain_entity=5, num_retain_relation=5, max_depth=5, max_iterations=5, score_threshold=0.8, exploration_constant=0.5):
        self.question = question
        # self.score_model = score_model
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.score_threshold = score_threshold
        self.alpha = 0.5  # Smoothing factor
        self.llm = llm
        self.num_retain_entity = num_retain_entity
        self.num_retain_relation = num_retain_relation
        self.score_method = 'none'
        self.exploration_constant = exploration_constant  # Add this line
        
        # Initialize root node, containing multiple topic entities
        entities_info = [{'entity_id': id, 'entity_name': name} 
                        for id, name in topic_entities.items()]
        self.root = MCTSNode(
            entities_info=entities_info,
            pre_relations=[],
            pre_head=-1
        )
        
    def search(self):
        """Execute the MCTS search process"""
        iterations = 0
        best_node = None
        best_value = float('-inf')
        
        while iterations < self.max_iterations:
            # Selection phase
            print(f"Iteration {iterations}:")
            node = self._select(self.root)
            if not node:
                break
            print(f"Selected node: {node.entities_info}, Cached Relations: {node.cached_relations}")
            
            # Expansion phase
            print('\nExpand start......')
            children = self._expand(node)
            print('\nExpand end......')
            if children:
                # Use value-based model for simulation
                if len(children) == 1 and children[0].v >= 0.9:
                    return self._extract_path(children[0])
                if self.score_method == 'vm':
                    best_child = max(children, key=lambda x: x.v)
                    simulated_v = self._simulate(best_child)
                    # Smoothly update the score of the best child node
                    best_child.v = best_child.v * (1 - self.alpha) + simulated_v * self.alpha
                    best_child.visits += 1
                
                # Backpropagation
                self._backpropagate(node)
                
                # Update global best node
                for child in children:
                    if child.v > best_value:
                        best_value = child.v
                        best_node = child
                    
                    # If a satisfactory solution is found, return directly
                    if self._check_solution(child):
                        return self._extract_path(child)
            
            iterations += 1
        
        # If no satisfactory solution is found, return the path of the best node in the search tree
        best_node = self._get_best_node() if best_node is None else best_node
        return self._extract_path(best_node) if best_node else []

    def evaluate(self, triples):
        question = self.question
        
        # Format each triple in the triples list as a string
        formatted_triples = []
        for triple in triples:
            # Triple format is (head_name, relation, tail_name)
            head_name, relation, tail_name = triple
            formatted_triple = f"{head_name}, {relation}, {tail_name}"
            formatted_triples.append(formatted_triple)
        
        # Join the formatted triples list into a multiline string
        triples_str = "\n".join(formatted_triples)
        
        # Generate formatted_prompt
        formatted_prompt = EVALUATE_STATE_PROMPT.format(question=question, triple=triples_str)
        value_eval = self.llm(formatted_prompt)[0]
        score_match = re.search(r"(\d+\.\d+)", value_eval)  # Match floating-point numbers
        if score_match:
            score = float(score_match.group(1))  # Extract the matched score and convert to float
        else:
            score = 0.0  # If no score is matched, return 0.0
        
        return score

    def entity_prune(self, candidate_entities, node, relation):
        # Get the names of candidate entities
        candidate_names = [get_entity_name(eid) for eid in candidate_entities]
        
        # Store the final matched entity IDs
        matched_entity_ids = []
        
        # Build a mapping from names to ID lists
        name_to_ids = defaultdict(list)
        for name, eid in zip(candidate_names, candidate_entities):
            name_to_ids[name].append(eid)
        
        # Get current entities and path history
        current_entities = ', '.join([current_entity['entity_name'] for current_entity in node.entities_info])
        path_history = ' -> '.join([f"{t[0]}-{t[1]}-{t[2]}" for t in node.y])
        
        # Call LLM to get the most relevant entity names
        llm_output = llm(entity_p_prompt.format(
            top_k=self.num_retain_entity,
            question=self.question,
            current_entities=current_entities,
            current_relation=relation,
            path_history=path_history,
            candidate_names=', '.join(candidate_names)
        ))[0]
        
        # Extract entity names
        entities = extract_entity_names(llm_output)
        
        # Iterate over entity names returned by LLM
        for name in entities:
            if name in name_to_ids and name_to_ids[name]:
                # Randomly select an ID
                selected_id = random.choice(name_to_ids[name])
                matched_entity_ids.append(selected_id)
                
                # Remove the selected ID from name_to_ids to avoid duplicate selection
                name_to_ids[name].remove(selected_id)
        
        return matched_entity_ids

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select an unexpanded node"""
        while node.is_fully_expanded and node.children:
            node = self._get_best_child(node)
            
        if self._is_terminal(node):
            return None
            
        return node
    
    def _expand(self, node: MCTSNode):
        """Expand the node: Expand relations for all entities in the current node and create child nodes for each possible relation"""
        children = []
        
        # Expand each entity
        for entity_idx, entity_info in enumerate(node.entities_info):
            entity_id = entity_info['entity_id']
            
            # Check if the entity has been expanded in previous nodes
            if entity_id in node.cached_relations:
                # If it has been fully expanded, skip the entity
                print('-------------------------------'*5)
                print(f'{entity_id} has been expanded')
                continue
                # if len(node.expanded_relations[entity_id]) == len(node.cached_relations[entity_id]):
                #     node.mark_entity_fully_expanded(entity_idx)
                #     continue
                # relations = node.cached_relations[entity_id]
            else:
                # If not cached, search and cache
                relations = relation_search_prune(
                    entity_id,
                    entity_info['entity_name'],
                    node.pre_relations,
                    node.pre_head,
                    self.question,
                    self.llm
                )
                node.cache_relations(entity_id, relations)
            
            # Create child nodes for each unexpanded relation
            for relation_info in relations:
                if relation_info['relation'] in node.expanded_relations[entity_id]:
                    continue
                    
                target_entities = entity_search(
                    entity_id,
                    relation_info['relation'],
                    relation_info['head']
                )
                
                if len(target_entities) >=50:
                    target_entities = random.sample(target_entities, 40)

                if len(target_entities) >=10:
                    # target_entities = random.sample(target_entities, self.num_retain_entity)
                    target_entities = self.entity_prune(target_entities, node, relation_info['relation'])
                    
                for target_id in target_entities:
                    target_name = get_entity_name(target_id)
                    
                    # Check if the target entity is already in the path
                    if any(target_id == e['entity_id'] for e in node.entities_info):
                        continue
                    
                    # Construct new triple
                    print(entity_info['entity_name'],relation_info['relation'], target_name, relation_info['head'])
                    new_triple = _construct_triple(entity_info['entity_name'],relation_info['relation'], target_name, relation_info['head'])
                    # new_triple = {
                    #     'subject': entity_info['entity_name'],
                    #     'relation': relation_info['relation'],
                    #     'object': target_name,
                    #     'head': relation_info['head']
                    # }
                    
                    # Create child node
                    child = node.add_child(
                        entity_idx=entity_idx,
                        new_entity_info={'entity_id': target_id, 'entity_name': target_name},
                        new_triple=new_triple,
                        relation=relation_info['relation'],
                        head=relation_info['head']
                    )
                    
                    # Evaluate child node
                    # finish it tmr
                    child.v = self.evaluate(child.y)
                    children.append(child)
                    if child.v >= 0.9:
                        return [child]
                    
                # Mark the relation as expanded
                node.expanded_relations[entity_id].add(relation_info['relation'])
            
            # Check if fully expanded
            if len(node.expanded_relations[entity_id]) == len(relations):
                node.mark_entity_fully_expanded(entity_idx)
        
        # Check if all entities are fully expanded
        if all(len(expanded) == len(node.get_cached_relations(entity['entity_id']))
            for entity, expanded in zip(node.entities_info, node.expanded_relations.values())):
            node.is_fully_expanded = True
        
        return children
    
    def _simulate(self, node: MCTSNode, roll_forward_steps: int = 3) -> float:
        """Random simulation"""
        max_value = node.v
        current_path = node.y.copy()
        
        # Randomly select an entity as the starting point
        entity_idx = random.randrange(len(node.entities_info))
        current_entity = node.entities_info[entity_idx]
        pre_relations = node.pre_relations.copy()
        pre_head = node.pre_head
        print('\nSimulation start......')
        for _ in range(roll_forward_steps):
            relations = relation_search_prune(
                current_entity['entity_id'],
                current_entity['entity_name'],
                pre_relations,
                pre_head,
                self.question,
                self.llm
            )
            
            if not relations:
                break
                
            relation_info = relations[0]
            target_entities = entity_search(
                current_entity['entity_id'],
                relation_info['relation'],
                relation_info['head']
            )
            
            if not target_entities:
                break
                
            target_id = target_entities[0]
            target_name = get_entity_name(target_id)
            
            new_triple = {
                'subject': current_entity['entity_name'],
                'relation': relation_info['relation'],
                'object': target_name,
                'head': relation_info['head']
            }
            
            current_path.append(new_triple)
            
            try:
                current_value = self.evaluate(self.question, current_path)
                max_value = max(max_value, current_value)
            except Exception as e:
                print(f"Evaluation error: {e}")
                break
                
            if current_value >= self.score_threshold:
                break
                
            current_entity = {'entity_id': target_id, 'entity_name': target_name}
            pre_relations.append(relation_info['relation'])
            pre_head = relation_info['head']

        print('Simulation end......\n')
        return max_value
        
    def _backpropagate(self, node: MCTSNode):
        """Backpropagate to update node statistics"""
        current = node
        while current is not None:
            current.visits += 1
            current = current.parent
            
    def _get_best_child(self, node: MCTSNode) -> MCTSNode:
        """Select the best child node using UCT"""
        return max(node.children, key=lambda x: x.get_uct_value(self.exploration_constant))
        
    def _is_terminal(self, node: MCTSNode) -> bool:
        """Check if it is a terminal node"""
        return len(node.y) >= self.max_depth or node.is_fully_expanded and not node.children
        
    def _check_solution(self, node: MCTSNode) -> bool:
        """Check if a satisfactory solution is found"""
        return node.v >= self.score_threshold and len(node.y) <= self.max_depth
        
    def _extract_path(self, node: MCTSNode):
        """Extract the path"""
        return node.y

    def _get_best_node(self):
        """Get the node with the highest score in the search tree
        
        Returns:
            MCTSNode: Node with the highest score
        """
        def get_best_v(node):
            """Recursively get the node with the highest score in the subtree
            
            Returns:
                Tuple[MCTSNode, float]: (Best node, Highest score)
            """
            if not node.children:  # If it is a leaf node
                return node, node.v
                
            max_v = node.v
            max_node = node
            
            for child in node.children:
                child_node, child_v = get_best_v(child)
                if child_v > max_v:
                    max_v = child_v
                    max_node = child_node
                    
            return max_node, max_v
        
        # Start searching from the root node
        best_node, _ = get_best_v(self.root)
        return best_node

    def _extract_path(self, node):
        """Extract the complete path from the node
        
        Args:
            node: MCTSNode Target node
            
        Returns:
            List[Dict]: Complete path from the root node to the target node
        """
        return node.y