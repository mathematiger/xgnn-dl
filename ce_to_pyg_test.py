# from create_random_ce import mutate_ce
from owlapy.class_expression import OWLClassExpression, OWLObjectUnionOf, OWLObjectCardinalityRestriction, OWLObjectMinCardinality
from owlapy.class_expression import OWLClass, OWLObjectIntersectionOf, OWLCardinalityRestriction, OWLNaryBooleanClassExpression, OWLObjectRestriction
from owlapy.owl_property import OWLObjectProperty
from owlapy.render import DLSyntaxObjectRenderer
import unittest
import sys
import os
from create_random_ce import Mutation, CEUtils
import copy
from ce_to_pyg import CreateHeteroData
from torch_geometric.data import HeteroData
import torch
dlsr = DLSyntaxObjectRenderer()
xmlns = "http://www.semanticweb.org/stefan/ontologies/2023/1/untitled-ontology-11#"
NS = xmlns

class_3 = OWLClass('#3')
class_2 = OWLClass('#2')
class_1 = OWLClass('#1')
class_0 = OWLClass('#0')
edge = OWLObjectProperty('#to')


class TestMutateCE(unittest.TestCase):
    def setUp(self):
        self.new_edge = OWLObjectMinCardinality(
            cardinality=1, filler=class_1, property=edge)
        self.ce_3_to_1 = OWLObjectIntersectionOf([class_3, OWLObjectMinCardinality(
            cardinality=1, filler=class_1, property=edge)])
        self.ce_1_to_1 = OWLObjectIntersectionOf([class_1, OWLObjectMinCardinality(
            cardinality=1, filler=class_1, property=edge)])
        self.ce_3_to_1OR2 = OWLObjectIntersectionOf([class_3, OWLObjectMinCardinality(
            cardinality=1, filler=OWLObjectUnionOf([class_1, class_2]), property=edge)])
        self.ce_3_to_1_to_3 = OWLObjectIntersectionOf([class_3, OWLObjectMinCardinality(
            cardinality=1,
            filler=OWLObjectIntersectionOf([class_1,
                                            OWLObjectMinCardinality(cardinality=1,
                                                                    filler=class_3,
                                                                    property=edge)]), property=edge)])

    def test_retrieve_edges(self):
        hdata_class = CreateHeteroData()
        edges, _ = hdata_class.retrieve_edges(self.ce_3_to_1)
        self.assertEqual(edges, [{('3', 'to', '1'): (0, 0)}])
        edges2, _ = hdata_class.retrieve_edges(self.ce_3_to_1_to_3)
        self.assertEqual(edges2, [{('3', 'to', '1'): (0, 0)}, {
                         ('1', 'to', '3'): (0, 1)}])

    def test_create_hetero_data_from_ce(self):
        hdata_class = CreateHeteroData()
        hdata_test_31 = HeteroData()
        hdata_test_31['3'] = torch.randn(1, 1)
        hdata_test_31['1'] = torch.randn(1, 1)
        hdata_test_31['3', 'to', '1'].edge_index = torch.tensor([[0], [0]])
        hdata_test_31['1', 'to', '3'].edge_index = torch.tensor([[0], [1]])
        hdata = hdata_class.create_hetero_data_from_ce(self.ce_3_to_1)
        self.assertEqual(hdata.edge_types, hdata_test_31.edge_types)
        hdata_test_313 = HeteroData()
        hdata_test_313['3'] = torch.randn(2, 1)
        hdata_test_313['1'] = torch.randn(2, 1)
        hdata_test_313['3', 'to', '1'].edge_index = torch.tensor(
            [[0, 1], [0, 0]])
        hdata_test_313['1', 'to', '3'].edge_index = torch.tensor(
            [[0, 0], [0, 1]])
        hdata = hdata_class.create_hetero_data_from_ce(self.ce_3_to_1_to_3)
        self.assertEqual(hdata.edge_types, hdata_test_313.edge_types)


if __name__ == '__main__':
    unittest.main()
