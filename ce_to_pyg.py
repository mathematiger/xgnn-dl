import torch
import random
from create_random_ce import CEUtils
from pyg_utils import PyGUtils
from torch_geometric.data import HeteroData
from owlapy.class_expression import OWLClassExpression, OWLObjectUnionOf, OWLObjectCardinalityRestriction, OWLObjectMinCardinality
from owlapy.class_expression import OWLClass, OWLObjectIntersectionOf, OWLCardinalityRestriction, OWLNaryBooleanClassExpression, OWLObjectRestriction
from owlapy.owl_property import OWLObjectProperty
from owlapy.render import DLSyntaxObjectRenderer
from owlapy.render import DLSyntaxObjectRenderer
dlsr = DLSyntaxObjectRenderer()


class CreateHeteroData():
    def __init__(self):
        pass

    @staticmethod
    def create_hetero_data_from_ce(ce):
        edges, nodetype_dict = CreateHeteroData.retrieve_edges(ce)
        hdata = HeteroData()
        for key, value in nodetype_dict.items():
            hdata[key].x = torch.randn(value, 1)
        for edge in edges:
            for key, value in edge.items():
                hdata = PyGUtils.add_edge_to_hdata(
                    hdata, key[0], key[1], key[2], value[0], value[1])
        return hdata

    @staticmethod
    def retrieve_edges(ce):
        edge_list_of_dict = []  # entries {(nt,edge,nt):(1,2)}
        nodetype_dict = {}

        def utils_find_id(nodetype, nodetype_dict):
            if nodetype not in nodetype_dict:
                nodetype_dict[nodetype] = 0
            else:
                nodetype_dict[nodetype] += 1
            return nodetype_dict[nodetype]

        def iterate_ce(ce, current_nodetype, current_id):
            if isinstance(ce, OWLObjectIntersectionOf):
                for op in ce.operands():
                    if isinstance(op, OWLClass):
                        if current_nodetype is None:
                            current_nodetype = str(dlsr.render(op))
                        if current_id is None:
                            current_id = utils_find_id(
                                current_nodetype, nodetype_dict)
                        break
                for op in ce.operands():
                    if not isinstance(op, OWLClass):
                        iterate_ce(op, current_nodetype, current_id)
            elif isinstance(ce, OWLObjectUnionOf):
                number_unions = CEUtils.number_unions(ce)
                random_number_unions = random.randint(1, number_unions)
                true_false_list = [True] * random_number_unions + \
                    [False] * (number_unions - random_number_unions)
                random.shuffle(true_false_list)
                possclasses = []
                for i, op in enumerate(ce.operands()):
                    if true_false_list[i]:
                        if isinstance(op, OWLClass):
                            possclasses.append(op)
                current_nodetype = str(dlsr.render(random.choice(possclasses)))
                current_id = utils_find_id(current_nodetype, nodetype_dict)
                for i, op in enumerate(ce.operands()):
                    if true_false_list[i]:
                        if not isinstance(op, OWLClass):
                            iterate_ce(op, current_nodetype, current_id)
            elif isinstance(ce, OWLClass):
                current_nodetype = str(dlsr.render(ce))
                current_id = utils_find_id(current_nodetype, nodetype_dict)
            elif isinstance(ce, OWLObjectMinCardinality):
                edge_type = str(dlsr.render(ce._property))
                num_edges = ce._cardinality
                endnode = CEUtils.return_nth_class(ce._filler, 1)
                if isinstance(endnode, OWLClass):
                    endnodetype = str(dlsr.render(endnode))
                else:
                    endnodetype = str(endnode)
                end_id = utils_find_id(endnodetype, nodetype_dict)
                edge_list_of_dict.append({(current_nodetype, edge_type, endnodetype): (
                    current_id, end_id)})
                if not isinstance(ce._filler, OWLClass):
                    print('debug', end_id)
                    iterate_ce(ce._filler, endnodetype,
                               end_id)
        iterate_ce(ce, None, None)
        return edge_list_of_dict, nodetype_dict
