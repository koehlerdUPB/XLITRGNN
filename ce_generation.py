
import numpy
import sys
import copy
import random
import torch
#sys.path.append('/Ontolearn')
#import generatingXgraphs


from ontolearn.concept_learner import CELOE
from ontolearn.model_adapter import ModelAdapter
from owlapy.model import OWLNamedIndividual, IRI
from owlapy.namespaces import Namespaces
from owlapy.render import DLSyntaxObjectRenderer
from examples.experiments_standard import ClosedWorld_ReasonerFactory
from owlapy.model import OWLObjectProperty, OWLObjectSomeValuesFrom
from owlapy.model import OWLDataProperty
from owlapy.model import OWLClass, OWLClassExpression
from owlapy.model import OWLDeclarationAxiom, OWLDatatype, OWLDataSomeValuesFrom, OWLObjectIntersectionOf, OWLEquivalentClassesAxiom, OWLObjectUnionOf
from owlapy.model import OWLDataPropertyDomainAxiom
from owlapy.model import IRI
from owlapy.owlready2 import OWLOntologyManager_Owlready2
from owlapy.render import DLSyntaxObjectRenderer
from ontolearn.core.owl.utils import OWLClassExpressionLengthMetric


random_seed = 3006
random.seed(random_seed)
dlsr = DLSyntaxObjectRenderer()
xmlns = "http://www.semanticweb.org/stefan/ontologies/2023/1/untitled-ontology-11#"
NS = xmlns

r_new = OWLDataProperty(IRI(NS, 'ER_new'+str(1)))
class_paper = OWLClass(IRI(NS, "Paper"))
obj_prop = OWLObjectProperty(IRI(NS, 'citesPaper'))
r_union = OWLObjectIntersectionOf([class_paper, r_new])
test_var = OWLObjectSomeValuesFrom(property=obj_prop, filler=r_union)






#here we load a dataset from the rest of the code



#generate lists from a class expression:

##utils
# remove the xmlns-part of the string
def remove_front(s):
    if len(s) == 0:
        return s
    else:
        return s[len(xmlns):]

    
def remove_end(s, list_of_suffix):
    suff = 'None'
    for suffix in list_of_suffix:
        if s.endswith(suffix):
            suff = suffix
            return s[:len(s)-len(suffix)], suff
    print('no matching end found')
    return s, suff


def get_edge_node_types(s, list_of_nodetypes):
    edge_type = remove_front(s)
    edge_type, node_type = remove_end(edge_type, list_of_nodetypes)
    return edge_type, node_type


def readout_OWLclass(ce: OWLClassExpression, list_nodetypes = []):
    #if callable(ce.operands):
    if isinstance(ce, OWLObjectIntersectionOf) or isinstance(ce, OWLObjectUnionOf):
        for op in ce.operands():
                if isinstance(op, OWLClass):
                    list_nodetypes.append(remove_front(op.to_string_id()))
                else:
                    list_node_types = readout_OWLclass(op, list_nodetypes)
                #elif callable(op.operands):
                #    list_node_types = readout_OWLclass(op, list_nodetypes)
    elif isinstance(ce, OWLClass):
        list_nodetypes.append(remove_front(ce.to_string_id()))
    set_nodetypes = set(list_nodetypes)
    return list(set_nodetypes)


def length_ce(ce):
    length_metric = OWLClassExpressionLengthMetric.get_default()
    return length_metric.length(ce)


def return_available_edgetypes(list_of_class_objs, search_class):
    for sublist in list_of_class_objs:
        if sublist[0] == search_class:
            second_argument = sublist[1]
            break
        else:
            # Handle case where the first argument is not found
            second_argument = None
    return second_argument
def return_available_classes(list_of_class_objs, search_edge):
    return_list = []
    for sublist in list_of_class_objs:
        if search_edge in sublist[1]:
            return_list.append(sublist[1])
    return return_list
# initialize and create the 3 lists
#example list: [0, [0], [1,[a,b]],[[sample_string, 0]]]






# ------ testing function --
def create_test_ce():
    class_3 = OWLClass(IRI(NS,'3'))
    class_2 = OWLClass(IRI(NS,'2'))
    class_1 = OWLClass(IRI(NS,'1'))
    class_0 = OWLClass(IRI(NS,'0'))
    edge = OWLObjectProperty(IRI(NS, 'to'))
    #CE 3-2-1
    edge_end = OWLObjectSomeValuesFrom(property=edge, filler=class_1)
    filler_end = OWLObjectIntersectionOf([class_2, edge_end])
    edge_middle = OWLObjectSomeValuesFrom(property=edge, filler=filler_end)
    ce_321 = OWLObjectIntersectionOf([class_3, edge_middle])
    print(128, 'ce', dlsr.render(ce_321))
    
    
    ce_3011 = OWLObjectIntersectionOf([class_3, OWLObjectSomeValuesFrom(property=edge, filler=OWLObjectIntersectionOf([class_0, OWLObjectSomeValuesFrom(property=edge, filler=OWLObjectIntersectionOf([class_1, OWLObjectSomeValuesFrom(property=edge, filler=class_1)]))]))])
    return generate_cedict_from_ce(ce_321), ce_3011











def update_class(ce: OWLClassExpression, list_result = [{'id' : 0, 'edge_types' : [], 'data_prop' : [], 'class_type' : [] }],  current_id = -1, current_class = '', current_mp_id = -1, current_result = list(), dict_class_ids = dict()):
    if isinstance(ce, OWLClass):
        if current_mp_id == -1:
            current_class = remove_front(ce.to_string_id())
            current_id = 0
            dict_class_ids[current_class] = [current_id]
        else:
            if current_result[current_mp_id][0][2] == '':
                current_class = remove_front(ce.to_string_id())
                current_result[current_mp_id][0][2] = current_class
                #choose random number to decide whether to add edge to an existent node or to a new node
                new_or_old_node = random.randint(0, 1)
                if current_class == current_result[current_mp_id][0][0]:
                    current_id = len(dict_class_ids[current_class])
                    dict_class_ids[current_class].append(current_id)
                elif new_or_old_node == 0 and current_class in dict_class_ids:  #added to an old node, if 0
                    current_id = random.randint(0, dict_class_ids[current_class][-1])  
                else:   #create new node
                    if current_class in dict_class_ids:
                        current_id =  len(dict_class_ids[current_class])
                        dict_class_ids[current_class].append(current_id)
                    else:
                        current_id = 0
                        dict_class_ids[current_class] = [current_id]
                #add current id
                current_result[current_mp_id][1][1] = [current_id]
    return current_class, current_id, current_result, dict_class_ids







def generate_cedict_from_ce(ce: OWLClassExpression, list_result = [{'id' : 0, 'edge_types' : [], 'data_prop' : [], 'class_type' : [] }],  current_id = -1, current_class = '', current_mp_id = -1, current_result = list(), dict_class_ids = dict(), choose_also_old_nodes = True):
    #idea: for each edge, we create a new entry [start,edge,end] : [tensor1, tensor2]; then, we map these tensors into one dict which describes a graph in PyG
    #mp = [[startnode, edge, endnode], [tensor 1, tensor 2]]
    if isinstance(ce, OWLClass):
        #here the last ce is class and should be added
        current_result[current_mp_id][0][2]=remove_front(ce.to_string_id())
        current_class = remove_front(ce.to_string_id())
        new_or_old_node = 1
        if choose_also_old_nodes:
            new_or_old_node = random.randint(0, 1)
        if current_class == current_result[current_mp_id][0][0]:
            current_id = len(dict_class_ids[current_class])
            dict_class_ids[current_class].append(current_id)
        elif new_or_old_node == 0 and current_class in dict_class_ids:  #added to an old node, if 0
            current_id = random.randint(0, dict_class_ids[current_class][-1])  
        else:   #create new node
            if current_class in dict_class_ids:
                current_id =  len(dict_class_ids[current_class])
                dict_class_ids[current_class].append(current_id)
            else:
                current_id = 0
                dict_class_ids[current_class] = [current_id]
        current_result[current_mp_id][1][1] = [current_id]
        #print('here should be something implemented')
    elif isinstance(ce, OWLObjectProperty): #append an edge, but without an endnode-type
        new_mp = [[current_class, remove_front(ce.to_string_id()), ''], [[current_id],[]]]
        current_result.append(new_mp)
        current_mp_id = len(current_result)
        # TODO: Map the end-node to some valid thing
    elif isinstance(ce, OWLDataProperty):
        print('164: DataProperties are not implemented yet')
    elif isinstance(ce, OWLObjectSomeValuesFrom):
        new_edge = remove_front(ce._property.to_string_id())
        new_mp = [[current_class, new_edge, ''], [[current_id],[]]]
        current_mp_id = len(current_result)
        current_result.append(new_mp)
        #iterate over filler
        generate_cedict_from_ce(ce._filler, current_id = current_id, current_class = current_class, current_mp_id = current_mp_id, current_result = current_result, dict_class_ids = dict_class_ids)
    elif isinstance(ce, OWLObjectIntersectionOf):
        op_classes = list()
        #cii = copy.deepcopy(current_id)
        #cci = copy.deepcopy(current_class)
        #cmi = copy.deepcopy(current_mp_id)
        for op in ce.operands():
            if isinstance(op, OWLClass):
                op_classes.append(op)
                current_class, current_id, current_result, dict_class_ids = update_class(op, current_id = current_id, current_class = current_class, current_mp_id = current_mp_id, current_result = current_result, dict_class_ids = dict_class_ids)
                #generate_cedict_from_ce(op, current_id = current_id, current_class = current_class, current_mp_id = current_mp_id, current_result = current_result)
        for op in ce.operands():
            if op not in op_classes:
                generate_cedict_from_ce(op, current_id = current_id, current_class = current_class, current_mp_id = current_mp_id, current_result = current_result, dict_class_ids = dict_class_ids)
    elif isinstance(ce, OWLObjectUnionOf):
        list_helpind = []
        for op in ce.operands():
            list_helpind.append(op)
        number_of_attributes = random.randint(1, len(list_helpind))   #random but not implemented yet
        attributes_to_add = random.sample(list_helpind, number_of_attributes)
        for op in attributes_to_add:
            generate_cedict_from_ce(op, current_id = current_id, current_class = current_class, current_mp_id = current_mp_id, current_result = current_result, dict_class_ids = dict_class_ids)
    return current_result, dict_class_ids



def create_graphdict_from_cedict(ce_dict, list_of_node_types, list_of_edge_types, metagraph, dict_class_ids):
    # fill the gaps
    # note: Currently, only at the ends of CEs there are missing classes; update this section, if several nodes without class may be created
    for mp in ce_dict:
        if mp[0][2] == '':
            #take an available class
            avail_classes = list()
            new_class = ''
            for p in metagraph:
                if p[0] == mp[0][0] and p[1] == mp[0][1]:
                    avail_classes.append(p[2])
            new_class = random.choice(avail_classes)
            if new_class in dict_class_ids:
                new_or_old_id = random.randint(0, 1)
                if new_or_old_id == 0:
                    new_id = random.choice(dict_class_ids[new_class])
                else:
                    new_id = len(dict_class_ids[new_class])
            else:
                new_id = 0
            if new_class == '':
                print(228, 'no available edge; this should not happen in the current implementation')
            mp[0][2] = new_class
            mp[1][1] = [new_id]    
        if mp[0][0] == '':
            print(233, 'this should not happen with the current implementation')       
    dict_graph = dict()
    #created_node_ids = dict() #store {'stored_ids' : [id], id : [class_type(string), new_value(int)], id2 : ...}
    #created_node_ids['stored_ids'] = []
    for mp in ce_dict:
        if (mp[0][0], mp[0][1], mp[0][2]) in dict_graph:
            tensor_start_end = dict_graph[(mp[0][0], mp[0][1], mp[0][2])]# += torch.tensor(mp[1][0])
            tensor_start = tensor_start_end[0]
            tensor_end = tensor_start_end[1]
            #check if this edge was already created
            checkpoint = 0
            pairs = list(zip(tensor_start.tolist(), tensor_end.tolist()))
            if(mp[1][0][0], mp[1][1][0]) in pairs:
                checkpoint = 1
            
            
            if checkpoint == 0:
                tensor_start = torch.cat((tensor_start, torch.tensor(mp[1][0], dtype = torch.long)),0)
                tensor_end = torch.cat((tensor_end, torch.tensor(mp[1][1], dtype = torch.long)),0)
                dict_graph.update({(mp[0][0], mp[0][1], mp[0][2]) : (tensor_start, tensor_end)})
                tensor_start_end2 = dict_graph[(mp[0][2], mp[0][1], mp[0][0])]
                tensor_start2 = tensor_start_end2[0]
                tensor_end2 = tensor_start_end2[1]
                tensor_start2 = torch.cat((tensor_start2, torch.tensor(mp[1][1], dtype = torch.long)),0)
                tensor_end2 = torch.cat((tensor_end2, torch.tensor(mp[1][0], dtype = torch.long)),0)
                dict_graph.update({(mp[0][2], mp[0][1], mp[0][0]) : (tensor_start2, tensor_end2)})
        else:
            #if mp[1][0] != mp[1][1]:
            if mp[0][0] != mp[0][2]:
                dict_graph[(mp[0][0], mp[0][1], mp[0][2])] = (torch.tensor(mp[1][0]), torch.tensor(mp[1][1]))
                dict_graph[(mp[0][2], mp[0][1], mp[0][0])] = (torch.tensor(mp[1][1]), torch.tensor(mp[1][0]))
            else:
                tensor_equal_front = torch.tensor(mp[1][0]+mp[1][1])
                tensor_equal_end = torch.tensor(mp[1][1]+mp[1][0])
                dict_graph[(mp[0][2], mp[0][1], mp[0][0])] = (tensor_equal_front, tensor_equal_end)
    return dict_graph
    
    

def create_graphdict_from_ce(ce: OWLClassExpression, list_of_node_types, list_of_edge_types, metagraph):
    ce_dict = dict()
    ce_dict, dict_class_ids = generate_cedict_from_ce(ce, current_result = [], dict_class_ids = dict())
    graph_dict = create_graphdict_from_cedict(ce_dict, list_of_node_types, list_of_edge_types, metagraph, dict_class_ids)
    return graph_dict
    




def create_random_ce(list_classes_objprops, root_class, num_iterate = 1):
    #Form: start: root_node; 
        # iterate: choose class to add edge; add edge + choose random if to add a type to the node or not and if yes, which type
        # always add new 'information' with 'and'; only later at random with and or 'or'
    #for iteration: Global variables
    total_num_created_nodes = 0
    list_all_edges = []
    for sublist in list_classes_objprops:
        for el in sublist[1]:
            if el not in list_all_edges:
                list_all_edges.append(el)
    #set_all_edges = set(list_all_edges)
    #list_all_edges = list(set(list_all_edges))
    #list_possible_expressions = ['add_edge', 'add_class', 'add_feature']
    list_possible_expressions = ['add_edge_from_node', 'add_additional_edge','add_class']
    list_possible_expr_wo_class =  copy.deepcopy(list_possible_expressions)
    list_possible_expr_wo_class.remove('add_class')
    list_possible_actions_just_add_edges = ['add_edge_from_node']
    list_properties = []
    current_filler = '' 
    new_node_filler = ''
    #first action: add edge
    current_node_has_class = False
    last_edge = ''
    current_classes = []
    
    
    # iterate from here, build from last node in graph to root_node
    # decide, if to conclude this path and to start new path or to continue
    
    #initialize: start_node
    
    #TODO: Iteration in extra function
    for n in range(num_iterate):
        #choose random, which expression to take, from add_edge, add_class, add_feature
        action = random.choice(list_possible_actions_just_add_edges)
        #choose random, if this is added with intersection or union
        inter_or_union = random.randint(0, 1) #0 for intersection, 1 for union
        if action == 'add_class':
            if inter_or_union == 0:
                if current_node_has_class:
                    action = list_possible_expr_wo_class[random.randint(0, len(list_possible_expr_wo_class)-1)]
                else:
                    #choose class
                    current_node_has_class = True
                    #check which classes are available for the last edge
                    if last_edge == '':
                        rnd_class = list_classes_objprops[random.randint(0, len(list_classes_objprops)-1)][0]
                    else:
                        avail_classes = []
                        for sublist in list_classes_objprops:
                            if last_edge in sublist[1]:
                                avail_classes.append(sublist[0])
                        rnd_class = avail_classes[random.randint(0, len(avail_classes)-1)]
                    current_classes.append(rnd_class)
                    #add to rest
                    if current_filler == '':
                        current_filler = rnd_class
                    else:
                        current_filler = OWLObjectIntersectionOf([rnd_class, current_filler])
            else:
                rnd_class = list_classes_objprops[random.randint(0, len(list_classes_objprops)-1)][0]
                current_node_has_class = True
                current_classes.append(rnd_class)
                if current_filler == '':
                    current_filler = rnd_class
                else:
                    current_filler = OWLObjectUnionOf([rnd_class, current_filler])
        if action == 'add_edge_from_node':
            #choose random edge from current node-type
            if current_filler == '':
                current_classes = [root_class]
            list_avail_edges = list()
            for sublist in list_classes_objprops:
                if sublist[0] in current_classes:
                    for el in sublist[1]:
                        if el not in list_avail_edges:
                            list_avail_edges.append(el)
            rnd_edge = random.choice(list_avail_edges)
                
            #filler
            obj_prop = rnd_edge[0]
            if current_filler == '':
                current_filler = rnd_edge[1]
                current_filler = OWLObjectSomeValuesFrom(property=obj_prop, filler=current_filler)
            else:
                filler_new = OWLObjectIntersectionOf([current_filler, rnd_edge[1]])
                current_filler = OWLObjectSomeValuesFrom(property=obj_prop, filler=filler_new)
            #delete all current info:
            current_classes = [rnd_edge[1]]
            last_edge = rnd_edge     
        if action == 'add_additional_edge':
            print('the above action is not implemented yet')
            n=-1
            #choose an available edge
            #do this in an external fct
        if action == 'add_feature':
            print('the above action is not implemented yet')
            n=-1
    list_avail_edges = []
    for sublist in list_classes_objprops:
        if sublist[0] == root_class:
            for el in sublist[1]:
                if el not in list_avail_edges:
                    list_avail_edges.append(el)
    pre_result = OWLObjectSomeValuesFrom(property=obj_prop, filler=current_filler)
    result = OWLObjectIntersectionOf([root_class, current_filler])
    '''
    # 1 edge:
    current_class = root_class
    #choose edge_type:
    list_available_edge_types = return_available_edgetypes(list_classes_objprops, current_class)
    new_edge = random.choice(list_available_edge_types)
    add_class = random.choice([0,1])
    if add_class:
        available_class_list = return_available_classes(list_classes_objprops, new_edge)
        new_class = random.choice(available_class_list)
    obj_prop = new_edge
    obj_filler = NS
    if add_class:
        obj_filler = new_class
        obj_class = OWLObjectSomeValuesFrom(property=obj_prop, filler=obj_filler)
    else:
        obj_class = obj_prop
    new_ce = OWLObjectIntersectionOf([root_class, obj_class])
    print(new_ce)
    '''
    #return current_filler
    #print(372, dlsr.render(result))
    return result
    

def create_random_ce_from_BAHetero(num_iter):
    class0 = OWLClass(IRI(NS, '0'))
    class1 = OWLClass(IRI(NS, '1'))
    class2 = OWLClass(IRI(NS, '2'))
    class3 = OWLClass(IRI(NS, '3'))
    edge_type = OWLObjectProperty(IRI(NS, 'to'))
    list_classes_objprops = [[class0, [edge_type]], [class1, [edge_type]], [class2, [edge_type]], [class3, [edge_type]]]
    rand_ce = create_random_ce(list_classes_objprops, class3, num_iter)
    return rand_ce


def create_random_ce_from_metagraph(num_iter, metagraph, root_class):
    node_types = []
    for metapath in metagraph:
        if metapath[0] not in node_types:
            node_types.append(metapath[0])
        if metapath[2] not in node_types:
            node_types.append(metapath[2])
    root_class_ce = OWLClass(IRI(NS, root_class))
    list_classes_objprops = []
    for nt in node_types:
        list_et_to_class = list()
        for mp in metagraph:
            if mp[0] == nt:
                if [OWLObjectProperty(IRI(NS, mp[1])),OWLClass(IRI(NS, mp[2])) ] not in list_et_to_class:
                    list_et_to_class.append([OWLObjectProperty(IRI(NS, mp[1])),OWLClass(IRI(NS, mp[2])) ])
            if mp[2] == nt:
                if [OWLObjectProperty(IRI(NS, mp[1])),OWLClass(IRI(NS, mp[0])) ] not in list_et_to_class:
                    list_et_to_class.append([OWLObjectProperty(IRI(NS, mp[1])),OWLClass(IRI(NS, mp[0])) ])
        list_classes_objprops.append([OWLClass(IRI(NS, nt)), list_et_to_class])
    return create_random_ce(list_classes_objprops, root_class_ce, num_iter)
            
    












    
    

#testing: many outputs
root_node_type = 'Author'









class_paper = OWLClass(IRI(NS, "Paper"))  #is class expression
class_author = OWLClass(IRI(NS,'Author')) #also class expression
class_paper2 =  OWLClass(IRI(NS, "Paper2"))
class_author2 = OWLClass(IRI(NS,'Author2'))
class_paper3 =  OWLClass(IRI(NS, "Paper3"))
class_author3 = OWLClass(IRI(NS,'Author3'))
class_paper4 =  OWLClass(IRI(NS, "Paper4"))
class_author4 = OWLClass(IRI(NS,'Author4'))
data_prop = OWLDataProperty(IRI(NS,'hasAge'))


#inter_ce = OWLObjectIntersectionOf([class_paper, class_author, data_prop])   #data_prop not propertly working atm
inter_ce = OWLObjectIntersectionOf([class_paper, class_author])
inter_ce2 = OWLObjectIntersectionOf([class_paper2, class_author2])
union_ce = OWLObjectUnionOf([class_paper, class_author])
union_ce2 = OWLObjectUnionOf([class_paper2, class_author2])
union_ce3 = OWLObjectUnionOf([class_paper3, class_author3])
union_ce4 = OWLObjectUnionOf([class_paper4, class_author4])
union_inter_inter2_ce = OWLObjectUnionOf([inter_ce, inter_ce2])
inter_union_union2_ce = OWLObjectIntersectionOf([union_ce, union_ce2])
union_union_union_2_ce = OWLObjectUnionOf([union_ce, union_ce2])
union_union_union_2_ce2 = OWLObjectUnionOf([union_ce3, union_ce4])
#print(class_expression_to_lists(inter_ce, root_node_type))
u_u3_u3 = OWLObjectUnionOf([union_union_union_2_ce, union_union_union_2_ce2])
obj_prop = OWLObjectProperty(IRI(NS, 'citesPaper'))
obj_filler = union_ce
obj_test = OWLObjectSomeValuesFrom(property=obj_prop, filler=obj_filler)
'''  
print(class_expression_to_lists(union_ce, root_node_type))
print('Now: union of 2 intersections')
print(class_expression_to_lists(union_inter_inter2_ce, root_node_type))
print('Now: Union of 2 Unions')
print(class_expression_to_lists(union_union_union_2_ce, root_node_type))

print('Now: Intersection of 2 unions')
print(class_expression_to_lists(inter_union_union2_ce, root_node_type))

print('Now: Union of 2 Unions of unions')

print(class_expression_to_lists(u_u3_u3, root_node_type))

print('Now: Object Properties' )
print(class_expression_to_lists(inter_cl_obj, root_node_type))



print('Now: Data Properties' )

print(class_expression_to_lists(inter_ce, root_node_type))


print('Now: ObjectSomeValuesFrom')

print(class_expression_to_lists(obj_test, root_node_type))
#test
for _ in range(10):
    ce = create_random_ce_from_BAHetero(3)
    print('created class expression',ce)
    print(class_expression_to_lists(ce, '3'))
'''    
    
    
'''
#test 23.05.generate_cedict_from_ce
print('520: generate_cedict_from_ce testing started')
preresult = generate_cedict_from_ce(obj_test)
print('560', preresult)
print( dlsr.render(obj_test))
graph_dict = create_graphdict_from_cedict(preresult, ['Paper', 'Author'], ['citesPaper'])
print('628', graph_dict)

'''