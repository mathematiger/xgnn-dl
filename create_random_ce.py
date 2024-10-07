
import sys
import copy
import random
import torch
import copy
import time

# sys.path.append('/Ontolearn')
# import generatingXgraphs

# sys.path.append('/Ontolearn')
# import generatingXgraphs
from owlapy.class_expression import OWLClassExpression, OWLObjectUnionOf, OWLObjectCardinalityRestriction, OWLObjectMinCardinality
from owlapy.class_expression import OWLClass, OWLObjectIntersectionOf, OWLCardinalityRestriction, OWLNaryBooleanClassExpression, OWLObjectRestriction
from owlapy.owl_property import OWLObjectProperty
from owlapy.render import DLSyntaxObjectRenderer


dlsr = DLSyntaxObjectRenderer()


#  ----------- Functions for manipulating CEs


class CEUtils():
    """ 
    Here, we gather all functions, which:
    - TODO: "flatten" a CE by removing unions in unions or intersecs in intersecs
    - get the edetypes or nodetypes of a property or class
    - return the n-th union/intersection/class/etc. of a class expression
    - replace the n-th union/intersection/class/etc. of a CE with a new CE
    """
    __slots__ = ()

    def __init__(self):
        pass

    @staticmethod
    def get_name_from_class_or_property(ce):
        if not isinstance(ce, OWLClass):
            if not isinstance(ce, OWLObjectProperty):
                print("ce must be of type OWLClass or OWLObjectProperty")
                return None
        return str(dlsr.render(ce))

    @staticmethod
    def count_classes(ce):
        count = 0
        if isinstance(ce, OWLClass):
            count += 1
        elif isinstance(ce, OWLNaryBooleanClassExpression):
            for op in ce.operands():
                count += CEUtils.count_classes(op)
        elif isinstance(ce, OWLObjectRestriction):
            count += CEUtils.count_classes(ce._filler)
        return count

    @staticmethod
    def count_existential_restrictions(ce):
        count = 0
        if isinstance(ce, OWLObjectRestriction):
            count += 1
            count += CEUtils.count_existential_restrictions(ce._filler)
        elif isinstance(ce, OWLNaryBooleanClassExpression):
            for op in ce.operands():
                count += CEUtils.count_existential_restrictions(op)
        return count

    @staticmethod
    def count_intersections(ce):
        count = 0
        if isinstance(ce, OWLObjectIntersectionOf):
            count += 1
            for op in ce.operands():
                count += CEUtils.count_intersections(op)
        elif isinstance(ce, OWLNaryBooleanClassExpression):
            for op in ce.operands():
                count += CEUtils.count_intersections(op)
        elif isinstance(ce, OWLObjectRestriction):
            count += CEUtils.count_intersections(ce._filler)
        return count

    @staticmethod
    def count_unions(ce):
        count = 0
        if isinstance(ce, OWLObjectUnionOf):
            count += 1
            for op in ce.operands():
                count += CEUtils.count_unions(op)
            return count
        elif isinstance(ce, OWLNaryBooleanClassExpression):
            for op in ce.operands():
                count += CEUtils.count_unions(op)
            return count
        elif isinstance(ce, OWLObjectRestriction):
            return CEUtils.count_unions(ce._filler)
        return count

    @staticmethod
    def return_nth_class(ce, n):
        if isinstance(ce, OWLClass):
            if n == 1:
                return ce
            else:
                return n-1
        elif isinstance(ce, OWLNaryBooleanClassExpression):
            for op in ce.operands():
                result = CEUtils.return_nth_class(op, n)
                if isinstance(result, OWLClass):
                    return result
            return False
        elif isinstance(ce, OWLObjectRestriction):
            return CEUtils.return_nth_class(ce._filler, n)
        return False

    @staticmethod
    def return_nth_restriction(ce, n):
        if isinstance(ce, OWLObjectRestriction):
            if n == 1:
                return ce
            else:
                return CEUtils.return_nth_restriction(ce._filler, n-1)
        elif isinstance(ce, OWLNaryBooleanClassExpression):
            for op in ce.operands():
                result = CEUtils.return_nth_restriction(op, n)
                if isinstance(result, OWLObjectRestriction):
                    return result
            return 0
        return 0

    @staticmethod
    def return_nth_intersection(ce, n):
        if isinstance(ce, OWLObjectIntersectionOf):
            if n == 1:
                return ce
            else:
                n -= 1  # TODO: test, ob 2 intersections in einer Intersection gefunden werden
                for op in ce.operands():
                    result = CEUtils.return_nth_intersection(op, n)
                    if isinstance(result, OWLObjectIntersectionOf):
                        return result
                return 0
        elif isinstance(ce, OWLNaryBooleanClassExpression):
            for op in ce.operands():
                result = CEUtils.return_nth_intersection(op, n)
                if isinstance(result, OWLObjectIntersectionOf):
                    return result
            return 0
        elif isinstance(ce, OWLObjectRestriction):
            return CEUtils.return_nth_intersection(ce._filler, n)
        return 0

    @staticmethod
    def return_nth_union(ce, n):
        if isinstance(ce, OWLObjectUnionOf):
            if n == 1:
                return ce
            else:
                n -= 1  # TODO: test, ob 2 intersections in einer Intersection gefunden werden
                for op in ce.operands():
                    result = CEUtils.return_nth_union(op, n)
                    if isinstance(result, OWLObjectUnionOf):
                        return result
                return 0
        elif isinstance(ce, OWLNaryBooleanClassExpression):
            for op in ce.operands():
                result = CEUtils.return_nth_union(op, n)
                if isinstance(result, OWLObjectUnionOf):
                    return result
            return 0
        elif isinstance(ce, OWLObjectRestriction):
            return CEUtils.return_nth_union(ce._filler, n)
        return 0

    @staticmethod
    def replace_nth_class(ce, n, newpropertyvalue, topce=None):
        """
        Adds sth to a class:
        class C -> C AND new_ce (new_ce should be a Cardinality restriction) 
        This only works, if C is already in a larger Class expression
        Not working, if ce is exactly one class
        """
        if isinstance(ce, OWLClass):
            if n == 1:
                if topce is None:
                    topce = OWLObjectIntersectionOf([ce])
                if isinstance(topce, OWLObjectIntersectionOf):
                    if isinstance(newpropertyvalue, OWLCardinalityRestriction):
                        for op in topce.operands():
                            if op == newpropertyvalue:
                                op._cardinality += 1
                                return True
                        topce._operands += (newpropertyvalue,)
                        return True
                    elif isinstance(newpropertyvalue, OWLObjectIntersectionOf):
                        topce._operands += (newpropertyvalue,)
                        return True
                    print('type', type(newpropertyvalue))
                    print('sth has not gone well')
                    raise ValueError('type of newpropertyvalue not supported')
                elif isinstance(topce, OWLObjectRestriction):
                    new_filler = OWLObjectIntersectionOf(
                        [ce, newpropertyvalue])
                    topce._filler = new_filler
                    return True
                elif isinstance(topce, OWLObjectUnionOf):
                    # remove class from operands
                    # add intersection([class, newpropertyvalue])
                    list_operands = list(topce._operands)
                    list_operands.remove(ce)
                    list_operands.append(OWLObjectIntersectionOf(
                        [ce, newpropertyvalue]))
                    topce._operands = tuple(list_operands)
                    return True
                else:
                    raise Exception("Can't replace class in class expression")

        elif isinstance(ce, OWLNaryBooleanClassExpression):
            for op in ce.operands():
                if CEUtils.replace_nth_class(op, n, newpropertyvalue, ce):
                    return True
                if isinstance(op, OWLClass):
                    n -= 1
        elif isinstance(ce, OWLObjectRestriction):
            return CEUtils.replace_nth_class(ce._filler, n, newpropertyvalue, ce)

    @staticmethod
    def increase_nth_existential_restriction(ce, n, increase=1):
        """
        A ex. E >=m B -> A ex. E >=m+1 B 
        """
        if isinstance(ce, OWLCardinalityRestriction):
            if n == 1:
                ce._cardinality += increase
                return True
            n -= 1
            return CEUtils.increase_nth_existential_restriction(ce._filler, n)
        elif isinstance(ce, OWLNaryBooleanClassExpression):
            for op in ce.operands():
                if CEUtils.increase_nth_existential_restriction(op, n):
                    return True

        else:
            return False

    @staticmethod
    def replace_nth_intersection(ce, newedge, n):
        """
        A ex. to >=n B -> A ex. to >=n+1 B

        return True, if successfull, else False
        """
        if isinstance(ce, OWLObjectIntersectionOf):
            if n == 1:
                if isinstance(newedge, OWLCardinalityRestriction):
                    for op in ce.operands():
                        if op == newedge:
                            op._cardinality += 1
                            return True
                ce._operands += (newedge,)
                return True
            n -= 1
            for op in ce.operands():
                CEUtils.replace_nth_intersection(op, newedge, n)
        elif isinstance(ce, OWLNaryBooleanClassExpression):
            for op in ce.operands():
                CEUtils.replace_nth_intersection(op, newedge, n)
        elif isinstance(ce, OWLObjectRestriction):
            CEUtils.replace_nth_intersection(ce._filler, newedge, n)
        return False

    @staticmethod
    def replace_nth_cl_w_union(ce, newclass, n, topce=None):
        """
        A -> A or B  (A, B classes)
        return True if successful, else False
        """
        if isinstance(ce, OWLClass):
            if n == 1:
                newunion = OWLObjectUnionOf([newclass, ce])
                new_operands = tuple(
                    x for x in topce._operands if x != ce)
                new_operands += (newunion,)
                topce._operands = new_operands
                return True
            n -= 1
        elif isinstance(ce, OWLNaryBooleanClassExpression):
            for op in ce.operands():
                if CEUtils.replace_nth_cl_w_union(op, newclass, n, ce):
                    return True
        elif isinstance(ce, OWLObjectRestriction):
            return CEUtils.replace_nth_cl_w_union(ce._filler, newclass, n, ce)
        return False

    @staticmethod
    def flatten_ce_class_first(ce):
        new_ce = copy.deepcopy(ce)

        def flatten_intersections(ce):
            if isinstance(ce, OWLObjectIntersectionOf):
                new_operands = []
                for op in ce.operands():
                    if isinstance(op, OWLObjectIntersectionOf):
                        new_operands += op.operands()
                    else:
                        new_operands.append(op)
                ce._operands = tuple(new_operands)
            elif isinstance(ce, OWLNaryBooleanClassExpression):
                new_operands = []
                for op in ce.operands():
                    new_operands.append(flatten_intersections(op))
                ce._operands = tuple(new_operands)
            elif isinstance(ce, OWLObjectRestriction):
                flatten_intersections(ce._filler)

        def flatten_unions(ce):
            if isinstance(ce, OWLObjectUnionOf):
                new_operands = []
                for op in ce.operands():
                    if isinstance(op, OWLObjectUnionOf):
                        new_operands += op.operands()
                    else:
                        new_operands.append(op)
                    ce._operands = tuple(new_operands)
            elif isinstance(ce, OWLNaryBooleanClassExpression):
                for op in ce.operands():
                    flatten_unions(op)
            elif isinstance(ce, OWLObjectRestriction):
                flatten_unions(ce._filler)

        def reorder_nary_first_classes(ce):
            if isinstance(ce, OWLNaryBooleanClassExpression):
                new_operands = []
                for op in ce.operands():
                    if isinstance(op, OWLClass):
                        new_operands.append(op)
                for op in ce.operands():
                    if not isinstance(op, OWLClass):
                        new_operands.append(op)
                        reorder_nary_first_classes(op)
                ce._operands = tuple(new_operands)
            elif isinstance(ce, OWLObjectRestriction):
                reorder_nary_first_classes(ce._filler)

        flatten_intersections(new_ce)
        flatten_unions(new_ce)
        reorder_nary_first_classes(new_ce)

        return new_ce

    @staticmethod
    def get_max_depth(ce):
        """
        the max_depth returns the max number of restriction (edges)
            from top-class to some bottom
        Hence, max_depth represents the GNN depth of GNNs with .. layers
        """
        def get_max_depth_helper(ce):
            depth = 0
            if isinstance(ce, OWLNaryBooleanClassExpression):
                for op in ce.operands():
                    depth = max(depth, get_max_depth_helper(op))
            elif isinstance(ce, OWLObjectRestriction):
                depth += 1
                depth += get_max_depth_helper(ce._filler)
            return depth
        return get_max_depth_helper(ce)

    @staticmethod
    def find_all_poosible_mutations(ce):
        def find_all_poosible_mutations_helper(ce):
            mutations = []
            if isinstance(ce, OWLObjectIntersectionOf):
                mutations += ['intersection']
                if len(set(mutations)) == 4:
                    return list(set(mutations))
                for op in ce.operands():
                    mutations += find_all_poosible_mutations_helper(op)
            elif isinstance(ce, OWLObjectUnionOf):
                mutations += ['union']
                if len(set(mutations)) == 4:
                    return list(set(mutations))
                for op in ce.operands():
                    mutations += find_all_poosible_mutations_helper(op)
            elif isinstance(ce, OWLObjectRestriction):
                mutations += ['cardinality']
                if len(set(mutations)) == 4:
                    return list(set(mutations))
                mutations += find_all_poosible_mutations_helper(ce._filler)
            elif isinstance(ce, OWLClass):
                mutations += ['class']
                if len(set(mutations)) == 4:
                    return list(set(mutations))
            return list(set(mutations))
        return find_all_poosible_mutations_helper(ce)


# ----- mutate ce functions -----


class Mutation:
    def __init__(self, list_of_classes, list_of_edgetypes, max_depth=None):
        if isinstance(list_of_classes, list):
            for count, cls_ in enumerate(list_of_classes):
                if not isinstance(cls_, OWLClass):
                    cls_str = str(cls_)
                    list_of_classes[count] = OWLClass('#'+cls_str)
            self.list_of_classes = list_of_classes
        elif isinstance(list_of_classes, OWLClass):
            self.list_of_classes = [list_of_classes]
        else:
            raise Exception("list_of_classes is not a list or OWLClass")
        if isinstance(list_of_edgetypes, list):
            for count, edge in enumerate(list_of_edgetypes):
                if not isinstance(edge, OWLObjectProperty):
                    str_edge = str(edge)
                    list_of_edgetypes[count] = OWLObjectProperty('#'+str_edge)
            self.list_of_edgetypes = list_of_edgetypes
        elif isinstance(list_of_edgetypes, OWLObjectProperty):
            self.list_of_edgetypes = [list_of_edgetypes]
        else:
            raise Exception(
                "list_of_edge_types is not a list or OWLObjectProperty")
        if max_depth is not None:
            assert isinstance(max_depth, int), "max_depth must be an integer"
            self.max_depth = max_depth
        else:
            self.max_depth = 4

    def new_class(self):
        """
        return a new class
        """
        return random.choice(self.list_of_classes)

    def new_property(self):
        return random.choice(self.list_of_edgetypes)

    def new_edge(self):
        """
        return a new edge (min. cardinality restriction)
        """
        return OWLObjectMinCardinality(property=self.new_property(),
                                       cardinality=1,
                                       filler=self.new_class())

    def mutate_global(self, ce):
        """
        This fct takes a CE and mutates it by performing a random action:
        - add an intersection to a class: replace_nth_class
        - add a union to a class: replace_nth_cl_w_union
        - add an edge to an intersection: replace_nth_intersection
        - increase a cardinality restriction by 1:
            increase_nth_existential_restriction

        If a mutation returns False, it means mutation is not possible 
            -> Try a new mutation
        As we mutate CEs with at least one class, sth must work
        """
        possible_mutations = CEUtils.find_all_poosible_mutations(ce)

        def random_mutation(mutation, ce):

            if mutation == 'class':
                total_classes = CEUtils.count_classes(ce)
                if total_classes == 1:
                    new_ce = OWLObjectIntersectionOf([ce])
                    return random_mutation('intersection', new_ce)
                list_indices = [i for i in range(total_classes)]
                random.shuffle(list_indices)
                for i in list_indices:
                    try_ce = copy.deepcopy(ce)
                    if CEUtils.replace_nth_class(try_ce, i, self.new_edge()):
                        if CEUtils.get_max_depth(try_ce) <= self.max_depth:
                            return try_ce
                return False

            elif mutation == 'intersection':
                total_intersections = CEUtils.count_intersections(ce)
                if total_intersections == 0:
                    return False
                list_indices = [i for i in range(
                    1, total_intersections+1)]
                random.shuffle(list_indices)
                for i in list_indices:
                    try_ce = copy.deepcopy(ce)
                    if CEUtils.replace_nth_intersection(
                            try_ce, self.new_edge(), i):
                        if CEUtils.get_max_depth(try_ce) <= self.max_depth:
                            return try_ce
                return False

            elif mutation == 'union':
                total_unions = CEUtils.count_unions(ce)
                if total_unions == 0:
                    return False
                list_indices = [i for i in range(
                    1, total_unions)]
                random.shuffle(list_indices)
                for i in list_indices:
                    try_ce = copy.deepcopy(ce)
                    if CEUtils.replace_nth_cl_w_union(
                            try_ce, self.new_class(), i):
                        if CEUtils.get_max_depth(try_ce) <= self.max_depth:
                            return try_ce
                return False
            elif mutation == 'cardinality':
                total_restrictions = CEUtils.count_existential_restrictions(
                    ce)
                if total_restrictions == 0:
                    return False
                n = random.randint(1, total_restrictions)
                new_ce = copy.deepcopy(ce)
                if CEUtils.increase_nth_existential_restriction(new_ce, n):
                    return new_ce
            return False

        for mutation in possible_mutations:
            new_ce = random_mutation(mutation, ce)
            if new_ce is not False:
                return new_ce
        raise ValueError("Mutation not possible")

    def random_ce_with_startnode(self, length, typestart):
        assert isinstance(typestart, OWLClass)
        if length == 0:
            return typestart
        else:
            typestart = self.mutate_global(typestart)
            return self.random_ce_with_startnode(length-1, typestart)
