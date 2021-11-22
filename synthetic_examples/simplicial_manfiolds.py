import matplotlib, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import copy, itertools, traceback, networkx as nx, numpy as np
from scipy.spatial.distance import pdist, squareform
from networkx.drawing.nx_agraph import graphviz_layout

class Matrix():
    def __init__(self, x, is_frozen=False):
        self.__np_arr = Vector.to_real_nparray(x, dim=2)
        self.shape = self.__np_arr.shape
        self.is_frozen = is_frozen

    def freeze(self):
        self.is_frozen = True
        return self
    def copy(self, frozen=False): return Matrix(copy.deepcopy(self.__np_arr), is_frozen=frozen)
    def get_nparr(self): return copy.deepcopy(self.__np_arr)

    # Linear Algebra
    def T(self): return Matrix(self.__np_arr.T)
    def transpose(self): return self.T

    # Operators
    def __repr__(self): return f"Matrix({self.__np_arr.__repr__()})"
    def __str__(self): return str(self.__np_arr)
    def __getitem__(self, key):
        np_item = np.squeeze(self.__np_arr[key])
        if isinstance(np_item, np.ndarray):
            return Vector(np_item) if len(np_item.shape) == 1 else Matrix(np_item)
        else: return np_item
    def __iter__(self): return iter(copy.deepcopy(self.__np_arr))

    def __pos__(self): return Matrix(self.__np_arr)
    def __neg__(self): return Matrix(-self.__np_arr)

    def __operator_check(self, x, T, T_str, op, other_on_right=True):
        if isinstance(x, T): o = x
        else:
            try: o = T[0](x) if type(T) is tuple else T(x)
            except Exception as e:
                if r'{self}' not in op: op = "{self} " + f"{op}" + " {other}"

                op_str_val = op.format(self=self, other=x)
                op_str_typ = op.format(self="[self]", other="[other]")

                raise NotImplementedError(
                    f"Can't perform {op_str_typ}: {op_str_val}!"
                    f"Self is a vector, but other ({x}) is not & can't be coerced to a {T_str} ({e})!"
                )

        if isinstance(o, Matrix):
            # TODO: Brittle
            shapes = [self.shape, o.shape] if other_on_right else [o.shape, self.shape]
            shapes_neq = shapes[0] != shapes[1]

            if (op in ('+', '-') and shapes_neq) or (op in ('@',) and (shapes[0][-1] != shapes[-1][0])):
                    raise NotImplementedError(
                        f"Can't perform {op_str_typ}: {op_str_val}!"
                        f"Self is a {self.shape}-dim Matrix, but other ({o}) is a {o.shape}-dim Matrix!"
                    )
        elif isinstance(o, Vector):
            if self.shape[-1] != o.dim:
                raise NotImplementedError(
                    f"Can't perform {op_str_typ}: {op_str_val}!"
                    f"Self is a {self.shape}-dim Matrix, but other ({o}) is a {o.dim}-dim Vector!"
                )

        return o

    def __add__(self, other):
        other = self.__operator_check(other, Matrix, "matrix", "+")
        return Matrix(self.__np_arr + other.get_nparr())
    def __sub__(self, other):
        other = self.__operator_check(other, Matrix, "matrix", "-")
        return Matrix(self.__np_arr - other.get_nparr())
    def __iadd__(self, other):
        if self.is_frozen: raise NotImplementedError(f"Can't perform += while frozen!")
        other = self.__operator_check(other, Matrix, "matrix", "+=")
        self.__np_arr += other.get_nparr()
        return self
    def __isub__(self, other):
        if self.is_frozen: raise NotImplementedError(f"Can't perform -= while frozen!")
        other = self.__operator_check(other, Matrix, "matrix", "-=")
        self.__np_arr -= other.get_nparr()
        return self

    def __mul__(self, other):
        other = self.__operator_check(other, (float, int), "scalar", "*")
        return Matrix(self.__np_arr * other)
    def __truediv__(self, other):
        other = self.__operator_check(other, (float, int), "scalar", "/")
        return Matrix(self.__np_arr / other)
    def __rmul__(self, other):
        other = self.__operator_check(other, (float, int), "scalar", "*")
        return Matrix(self.__np_arr * other)
    def __rtruediv__(self, other):
        other = self.__operator_check(other, (float, int), "scalar", "/")
        return Matrix(self.__np_arr / other)
    def __imul__(self, other):
        if self.is_frozen: raise NotImplementedError(f"Can't perform *= while frozen!")
        other = self.__operator_check(other, (float, int), "scalar", "*=")
        self.__np_arr *= other
        return self
    def __itruediv__(self, other):
        if self.is_frozen: raise NotImplementedError(f"Can't perform /= while frozen!")
        other = self.__operator_check(other, (float, int), "scalar", "/=")
        self.__np_arr /= other
        return self

    def __matmul__(self, other):
        try:
            other = self.__operator_check(other, Vector, "Vector", "@", other_on_right=True)
            return Vector(np.around(self.__np_arr @ other.get_nparr(), decimals=12))
        except NotImplementedError as e_vector_sub:
            try:
                other = self.__operator_check(other, Matrix, "Matrix", "@", other_on_right=True)
                return Matrix(np.around(self.__np_arr @ other.get_nparr(), decimals=12))
            except NotImplementedError as e_matrix_sub:
                raise NotImplementedError(
                    "Matrix @ [other] expects [other] to either be a Vector or a Matrix.\n"
                    f"Provided: {type(self)} @ {type(other)}: {self} @ {other}.\n"
                    f"Failed to convert other to Vector: {e_vector_sub}.\n"
                    f"Failed to convert other to Matrix: {e_matrix_sub}."
                )

    def __imatmul__(self, other):
        if self.is_frozen: raise NotImplementedError(f"Can't perform @= while frozen!")
        other = self.__operator_check(other, Matrix, "matrix", "@", other_on_right=True)

        self.__np_arr = self.__np_arr @ other.get_nparr()
        self.shape = self.__np_arr.shape

        return self

    def __eq__(self, other):
        other = self.__operator_check(other, Vector, "vector", "==")
        return (self.__np_arr == other.get_nparr()).all()
    def __ne__(self, other):
        other = self.__operator_check(other, Vector, "vector", "!=")
        return (self.__np_arr == other.get_nparr()).all()

class Vector():
    @staticmethod
    def to_real_nparray(x, dim=1):
        x = copy.deepcopy(x)

        assert isinstance(dim, (int, float)) and dim > 0 and int(dim) == dim

        if not isinstance(x, np.ndarray):
            if dim == 1 and isinstance(x, Vector):
                x = x.get_nparr()
            else:
                assert isinstance(x, (list, tuple)), f"{x} must be an ordered sequence (is {type(x)}!"
                if dim == 1: x = np.array(x)
                else: x = np.array([Vector.to_real_nparray(e, dim=dim-1) for e in x])

        assert len(x.shape) == dim, f"{x} must have dim {dim}! Got shape: {x.shape}"

        assert np.issubdtype(x.dtype, np.number), f"{x} contains non-numerics!"
        assert np.isreal(x).all(), f"{x} contains complex numbers!"

        x = x.astype(float)
        assert not np.isnan(x).any(), f"{x} contains NaNs!"
        return x #np.around(x, decimals=6) # To handle small numerical errors

    def __init__(self, x, is_frozen=False):
        x = self.to_real_nparray(x, dim=1)
        self.dim = len(x)
        self.__np_arr = x
        self.is_frozen = is_frozen

    @classmethod
    def __convert(cls, *xs): return [x if isinstance(x, cls) else cls(x) for x in xs]

    @classmethod
    def dist(cls, x1, x2):
        try:
            x1, x2 = cls.__convert(x1, x2)
            return x1.distance_to(x2)
        except Exception as e:
            vec, hypersurface = None, None
            if isinstance(x1, Vector): vec = x1
            elif isinstance(x1, Hypersurface): hypersurface = x1
            else:
                try: vec = Vector(x1)
                except Exception as e1:
                    try: hypersurface = Hypersurface(*x1)
                    except Exception as e2:
                        raise NotImplementedError(
                            f"x1 ({x1}) is neither Vector nor Hypersurface!\n{e1}\n{e2}"
                        )

            if vec is None:
                if isinstance(x2, Vector):
                    vec = x2
                elif isinstance(x2, Hypersurface):
                    raise NotImplementedError(f"One of x1 or x2 ({x1} or {x2}) must be a Vector!")
                else:
                    try: vec = Vector(x2)
                    except Exception as e1:
                        raise NotImplementedError(
                            f"x1 is a hypersurface but x2 is not a Vector!\n{x1}\n{x2}\n{e1}"
                        )
            elif hypersurface is None:
                if isinstance(x2, Hypersurface):
                    hypersurface = x2
                elif isinstance(x2, Vector):
                    raise NotImplementedError(f"One of x1 or x2 ({x1} or {x2}) must be a Hypersurface!")
                else:
                    try: hypersurface = Hypersurface(*x2)
                    except Exception as e1:
                        raise NotImplementedError(
                            f"x1 is a Vector but x2 is not a Hypersurface!\n{x1}\n{x2}\n{e1}"
                        )

            assert vec is not None and hypersurface is not None
            return vec.distance_to(hypersurface)

    @classmethod
    def inner_product(cls, x1, x2):
        x1, x2 = cls.__convert(x1, x2)
        return x1 @ x2
    @classmethod
    def cos_between(cls, x1, x2):
        x1, x2 = cls.__convert(x1, x2)
        return cls.inner_product(x1, x2) / (x1.norm() * x2.norm())
    @classmethod
    def angle_between(cls, x1, x2):
        """Returns angle between x1 and x2 in radians"""
        return np.arccos(cls.cos_between(x1, x2))

    def freeze(self):
        self.is_frozen = True
        return self
    def copy(self, frozen=False): return Vector(copy.deepcopy(self.__np_arr), is_frozen=frozen)
    def get_nparr(self): return copy.deepcopy(self.__np_arr)

    # Linear Algebra
    def norm(self): return np.linalg.norm(self.__np_arr)
    def norm_sq(self): return sum(e**2 for e in self.__np_arr)
    def dot(self, other): return self @ other

    # Geometry
    def project_onto(self, other): return self | other
    def reflect_over(self, other): return self & other
    def distance_to(self, other):
        if isinstance(other, Vector):
            return np.around((self - other).norm(), decimals=12)
        elif isinstance(other, Hypersurface):
            return np.around((self - self.project_onto(other)).norm(), decimals=12)
        else:
            return Vector.dist(self, other) # This will try more aggressive type coercion

    # Operators
    def __repr__(self): return f"Vector({self.__np_arr.__repr__()})"
    def __str__(self): return str(self.__np_arr)
    def __hash__(self):
        assert self.is_frozen
        return tuple(self.__np_arr).__hash__()
    def __getitem__(self, key): return self.__np_arr[key]
    def __iter__(self): return iter(copy.deepcopy(self.__np_arr))

    def __pos__(self): return Vector(self.__np_arr)
    def __neg__(self): return Vector(-self.__np_arr)
    def __abs__(self): return self.norm()

    def __operator_check(self, x, T, T_str, op):
        if isinstance(x, T): o = x
        else:
            try: o = T[0](x) if type(T) is tuple else T(x)
            except Exception as e:
                if r'{self}' not in op: op = "{self} " + f"{op}" + " {other}"

                op_str_val = op.format(self=self, other=x)
                op_str_typ = op.format(self="[self]", other="[other]")

                raise NotImplementedError(
                    f"Can't perform {op_str_typ}: {op_str_val}!"
                    f"Self is a vector, but other ({x}) is not & can't be coerced to a {T_str} ({e})!"
                )

        if isinstance(o, Vector):
            if self.dim != o.dim:
                raise NotImplementedError(
                    f"Can't perform {op_str_typ}: {op_str_val}!"
                    f"Self is a {self.dim}-dim vector, but other ({o}) is a {o.dim}-dim vector!"
                )

        return o

    def __add__(self, other):
        other = self.__operator_check(other, Vector, "vector", "+")
        return Vector(self.__np_arr + other.get_nparr())
    def __sub__(self, other):
        try:
            other = self.__operator_check(other, Vector, "vector", "-")
            return Vector(self.__np_arr - other.get_nparr())
        except NotImplementedError as e_vector_sub:
            try:
                other = self.__operator_check(other, Hypersurface, "Hypersurface", "-")
                return Vector(self.__np_arr - (self | other).get_nparr())
            except NotImplementedError as e_hypersurface_sub:
                raise NotImplementedError(
                    "Vector - [other] expects [other] to either be a Vector or a Hypersurface.\n"
                    f"Provided: {type(self)} - {type(other)}: {self} - {other}.\n"
                    f"Failed to convert other to Vector: {e_vector_sub}.\n"
                    f"Failed to convert other to Hypersurace: {e_hypersurface_sub}."
                )

    def __iadd__(self, other):
        if self.is_frozen: raise NotImplementedError(f"Can't perform @= while frozen!")
        other = self.__operator_check(other, Vector, "vector", "+=")
        self.__np_arr += other.get_nparr()
        return self
    def __isub__(self, other):
        if self.is_frozen: raise NotImplementedError(f"Can't perform @= while frozen!")

        sub_result = self - other
        self.__np_arr = copy.deepcopy(sub_result.get_nparr())
        return self

    def __mul__(self, other):
        other = self.__operator_check(other, (float, int), "scalar", "*")
        return Vector(self.__np_arr * other)
    def __truediv__(self, other):
        other = self.__operator_check(other, (float, int), "scalar", "/")
        return Vector(self.__np_arr / other)
    def __rmul__(self, other):
        other = self.__operator_check(other, (float, int), "scalar", "*")
        return Vector(self.__np_arr * other)
    def __rtruediv__(self, other):
        other = self.__operator_check(other, (float, int), "scalar", "/")
        return Vector(self.__np_arr / other)
    def __imul__(self, other):
        if self.is_frozen: raise NotImplementedError(f"Can't perform @= while frozen!")
        other = self.__operator_check(other, (float, int), "scalar", "*=")
        self.__np_arr *= other
        return self
    def __itruediv__(self, other):
        if self.is_frozen: raise NotImplementedError(f"Can't perform @= while frozen!")
        other = self.__operator_check(other, (float, int), "scalar", "/=")
        self.__np_arr /= other
        return self

    def __matmul__(self, other):
        other = self.__operator_check(other, Vector, "vector", "<{self}, {other}>")
        return np.dot(self.__np_arr, other.get_nparr())

    # Projecting onto hypersurfaces
    def __or__(self, other):
        """
        We use | to signify projection. So if $v$ is a vector and $H$ a hypersurface, $v | H$ (like in math
        how | can mean restricted to) this means v projected to H.
        """
        try:
            other = self.__operator_check(other, Vector, "Vector", "proj_other(self) [|]")
            return (self.dot(other) / other.norm_sq()) * other
        except NotImplementedError as e_vec:
            try:
                other = self.__operator_check(other, Hypersurface, "Hypersurface", "proj_other(self) [|]")
                return other.proj_vector_onto_self_surface(self)
            except NotImplementedError as e_surf:
                raise NotImplementedError(f"{e_vec}\n{e_surf}")
    def __and__(self, other):
        """
        We use & to signify reflection. So if $v$ is a vector and $H$ a hypersurface, $v & H$ (there is no
        intuition for this one) this means v reflected over H (e.g., v + 2*((v|H) - v))
        """
        proj = self | other
        return self + 2*(proj - self)

    def __eq__(self, other):
        other = self.__operator_check(other, Vector, "vector", "==")
        return (self.__np_arr == other.get_nparr()).all()
    def __ne__(self, other):
        other = self.__operator_check(other, Vector, "vector", "!=")
        return (self.__np_arr == other.get_nparr()).all()

class Hypersurface():
    """
    This somewhat plays double duty --- it both reflects a simplex defined by the passed points
    and the full hypersurface implied by that simplex
    (e.g., the simplex is a triangle, the full surface is the plane).
    """
    def __init__(self, *xs):
        typed_xs = []
        for x in xs:
            if isinstance(x, Vector): x = x.copy(frozen=True)
            else: x = Vector(x, is_frozen=True)
            typed_xs.append(x)

        x0_dim = typed_xs[0].dim
        for x in typed_xs: assert x.dim == x0_dim

        self.space_dim = x0_dim
        assert len(typed_xs) <= self.space_dim

        self.surface_dim = len(typed_xs) - 1
        self.dim = self.surface_dim

        self.anchors = tuple(typed_xs)

        basis_vectors = []
        self.origin = self.anchors[0]
        for x in self.anchors[1:]:
            delta = x - self.origin
            for bv in basis_vectors:
                delta -= delta.project_onto(bv)

            assert delta.norm() > 0, f"Points are linearly dependent!"

            delta /= delta.norm()
            delta.freeze()
            basis_vectors.append(delta)

        self.basis_vectors = tuple(basis_vectors)
        if self.basis_vectors:
            self.M_coord_to_basis = Matrix(self.basis_vectors).T()

            self.anchors_in_coord = tuple([
                self.proj_vector_onto_self_coord(p).freeze() for p in self.anchors
            ])
        else: assert self.surface_dim == 0

    def __getitem__(self, key): return self.anchors[key]
    def __iter__(self): return iter(self.anchors)
    def __str__(self): return f"Hypersurface([{', '.join(str(a) for a in self.anchors)}])"
    def __repr__(self): return f"Hypersurface([{', '.join(a.__repr__() for a in self.anchors)}])"

    def segments_intersect(self, other):
        """
        Returns
            <is_parallel>, <is_collinear>,
            (<endpoints_coincident>, <endpoints_coincident_interior>, <endpoints_coincident_endpoints>),
            <interiors_intersect>
        TODO: Need to return interiors intersect, endpoints intersect, segments are colinear.
        """
        if isinstance(other, Vector): other = Hypersurface(hyperedge)
        elif isinstance(other, (list, tuple)): other = Hypersurface(*hyperedge)
        else: assert isinstance(other, Hypersurface)

        assert self.surface_dim == 1, f"For now this is all we handle."
        assert self.surface_dim == other.surface_dim, f"For now we only handle line intersections."

        # Let's start by determining the angle between our two hypersurfaces.
        l1_st, l1_end = self.anchors
        l2_st, l2_end = other.anchors

        lines_cos_sim = Vector.cos_between(l1_end-l1_st, l2_end-l2_st)

        # Parallel Checking
        is_parallel = np.isclose(np.abs(lines_cos_sim), 1)

        # Endpoint on line checking
        # To do this, first we need to figure out what the distance is between endpoints and lines.
        l1_st_proj_other  = l1_st  | other
        l1_end_proj_other = l1_end | other
        l2_st_proj_self   = l2_st  | self
        l2_end_proj_self  = l2_end | self

        l1_st_delta_other  = l1_st  - l1_st_proj_other
        l1_end_delta_other = l1_end - l1_end_proj_other
        l2_st_delta_self   = l2_st  - l2_st_proj_self
        l2_end_delta_self  = l2_end - l2_end_proj_self

        # In one case, one endpoint of *Segment* lives on other.
        l1_st_dist_other  = np.around(l1_st_delta_other.norm(), decimals=7)
        l1_end_dist_other = np.around(l1_end_delta_other.norm(), decimals=7)
        l2_st_dist_self   = np.around(l2_st_delta_self.norm(), decimals=7)
        l2_end_dist_self  = np.around(l2_end_delta_self.norm(), decimals=7)

        l1_st_on_other  = (l1_st_dist_other == 0)
        l1_end_on_other = (l1_end_dist_other == 0)
        l2_st_on_self   = (l2_st_dist_self == 0)
        l2_end_on_self  = (l2_end_dist_self == 0)

        endpoints_coincident = False
        endpoints_coincident_interior = False
        endpoints_coincident_endpoints = False
        num_endpoints_coincident_endpoints = 0
        for endpoint, surface, coincident in (
            (l1_st_proj_other,  other, l1_st_on_other),
            (l1_end_proj_other, other, l1_end_on_other),
            (l2_st_proj_self,   self,  l2_st_on_self),
            (l2_end_proj_self,  self,  l2_end_on_self),
        ):
            if not coincident: continue

            endpoints_coincident = True

            assert surface.anchors_in_coord[0] == Vector([0]), f"Line doesn't look right! {surface}"

            endpoint_local = surface.global_to_local(endpoint)[0]
            surface_extent = sorted([a[0] for a in surface.anchors_in_coord])

            if surface_extent[0] < endpoint_local and endpoint_local < surface_extent[1]:
                endpoints_coincident_interior = True
            elif surface_extent[0] == endpoint_local or endpoint_local == surface_extent[1]:
                num_endpoints_coincident_endpoints += 1
                endpoints_coincident_endpoints = True

        is_collinear = (l1_st_on_other and l1_end_on_other) or (l2_st_on_self and l2_end_on_self)

        # To find out if the interiors intersect, we need different processing based on several cases:
        if is_collinear:
            if not (l2_st_on_self and l2_end_on_self and l1_st_on_other and l1_end_on_other):
                print("WARNING: Inconsistent Collinearity Check!")
                print(self, other)
                print(l1_st_dist_other, l1_end_dist_other, l2_st_dist_self, l2_end_dist_self)

            interiors_intersect = endpoints_coincident_interior or (num_endpoints_coincident_endpoints == 4)
        elif is_parallel:
            assert not endpoints_coincident, (
                "Shouldn't be parallel with only one coincident endpoint! Got\n"
                f"l1_st_dist_other = {l1_st_dist_other},\n"
                f"l1_end_dist_other = {l1_end_dist_other},\n"
                f"l2_st_dist_self = {l2_st_dist_self},\n"
                f"l2_end_dist_self = {l2_end_dist_self},\n"
            )

            # If they're parallel and not collinear, their interiors don't intersect.
            interiors_intersect = False
        elif endpoints_coincident:
            # If they aren't colinear and one endpoint is on another one of the lines, then the interiors
            # can't ever intersect.
            interiors_intersect = False
        else:
            assert l1_st_dist_other > 0 and l1_end_dist_other > 0, "We've already checked the endpoints!"
            assert l2_st_dist_self  > 0 and l2_end_dist_self  > 0, "We've already checked the endpoints!"

            # Here, we need to check if their intersection point is within the interior. To do so, we'll
            # check if the line *segment* self intersects with the *line* other, Then if the line *segment*
            # other intersects with the *line* self. If both are true, the two segments intersect -- if not,
            # they don't. We'll do this check by looking at the angle between the projection of the
            # line-segment endpoints onto the line. If the vectors are parallel, then the line segment lives
            # uniformly above or below the line. If they are antiparallel (these are the only two options)
            # then the line segment intersects the line.

            cos_sim_l1 = Vector.cos_between(l1_st_delta_other, l1_end_delta_other)
            cos_sim_l2 = Vector.cos_between(l2_st_delta_self, l2_end_delta_self)

            # As these lines (in our context) all exist within (planar) simplices, they should be co-planar,
            # and thus the orthogonal deltas from the lines to the pairs of endpoints should each be mutually
            # parallel or antiparallel (within a single pair of endpoints).
            #
            # Why? In essence, the lines' intersection, plus two endpoints
            # of the two lines all together form a plane, and the orthogonal lines off either line alone
            # in this plane must be parallel or antiparallel as they're both orthogonal to the same line.
            non_planar_error_message = (
                "Segments appear to be non-planar?\n"
                f"  self (l1)   = {str(self)} : {str(l1_st)}---{str(l1_end)}\n"
                f"  other (l2)  = {str(other)} : {str(l2_st)}---{str(l2_end)}\n"
                f"  l1_st  | l2 = {str(l1_st_proj_other)},  l1_st_delta  = {str(l1_st_delta_other)}\n"
                f"  l1_end | l2 = {str(l1_end_proj_other)}, l1_end_delta = {str(l1_end_delta_other)}\n"
                f"  l2_st  | l1 = {str(l2_st_proj_self)},  l2_st_delta  = {str(l2_st_delta_self)}\n"
                f"  l2_end | l1 = {str(l2_end_proj_self)}, l2_end_delta = {str(l2_end_delta_self)}\n"
                f"  cos(l1_st_delta, l1_end_delta) = {cos_sim_l1}"
                f"  cos(l2_st_delta, l2_end_delta) = {cos_sim_l2}"
            )
            np.testing.assert_almost_equal(np.abs(cos_sim_l1), 1, err_msg = non_planar_error_message)
            np.testing.assert_almost_equal(np.abs(cos_sim_l2), 1, err_msg = non_planar_error_message)

            # To check if the segments intersect the lines, we'll check if their endpoints are on the same or
            # opposite sides of the line, which is determinable based on whether their endpoints' orthogonal
            # projections are parallel (same side) or antiparallel (opposite sides, indicating an
            # intersection).

            # *Segment* self intersects *Line* other:
            segment_self_intersects_line_other = np.isclose(cos_sim_l1, -1)
            # *Line* self intersects *Segment* other:
            line_self_intersects_segment_other = np.isclose(cos_sim_l2, -1)

            interiors_intersect = segment_self_intersects_line_other and line_self_intersects_segment_other

        return (
            is_parallel, is_collinear,
            (endpoints_coincident, endpoints_coincident_interior, endpoints_coincident_endpoints),
            interiors_intersect
        )

    def drop(self, hyperedge):
        if isinstance(hyperedge, Vector): hyperedge = Hypersurface(hyperedge)
        elif isinstance(hyperedge, (list, tuple)): hyperedge = Hypersurface(*hyperedge)
        else: assert isinstance(hyperedge, Hypersurface)

        self_anchors, hyperedge_anchors = set(self.anchors), set(hyperedge.anchors)
        assert hyperedge_anchors.issubset(self_anchors) and len(hyperedge_anchors) != len(self_anchors)

        new_anchors = list(self_anchors - hyperedge_anchors)
        return Hypersurface(*new_anchors)

    # Coordinte Transformations
    def local_to_global(self, x):
        return self.origin + (self.M_coord_to_basis @ x)
    def global_to_local(self, x):
        return self.M_coord_to_basis.T() @ (x - self.origin)

    def proj_vector_onto_self_surface(self, x):
        return self.local_to_global(self.proj_vector_onto_self_coord(x))
    def proj_vector_onto_self_coord(self, x): return self.global_to_local(x)

#     def proj_vector_onto_self_coord(self, x):
#         assert isinstance(x, Vector) and x.dim == self.space_dim

#         centered = x - self.origin
#         coords = [centered.project_onto(bv).norm() for bv in self.basis_vectors]

#         return Vector(coords)

#     def proj_vector_onto_self_surface(self, x):
#         return self.local_to_global(self.proj_vector_onto_self_coord(x))

    def __ror__(self, x):
        if not isinstance(x, Vector): x = Vector(x)
        return self.proj_vector_onto_self_surface(x)

    def reflect_vector_over_self(self, x): return x.reflect_over(self)

    def reflect_over_face(self, hyperedge):
        # We want to reflect this over the edge hyperedge, keeping the other local coordinates preserved.
        reflected_anchors = {a: hyperedge.reflect_vector_over_self(a) for a in self.drop(hyperedge)}
        # Doing it this way ensures we update the positions of the new anchors in a consistent manner.
        new_anchors = [reflected_anchors[a] if a in reflected_anchors else a for a in self.anchors]
        return Hypersurface(*new_anchors)

    def __and__(self, other): return self.reflect_over_face(other)

    def plot(
        self,
        extra_points=[],
        extra_lines=[],
        extra_faces=[],
        **view_init_kwargs,
    ):
        assert self.space_dim in (2, 3), f"Can't plot in {self.space_dim}-dim."

        if extra_lines: assert self.space_dim >= 2
        if extra_faces: assert self.space_dim >= 3

        mins = [float('inf') for _ in range(self.space_dim)]
        maxes = [-float('inf') for _ in range(self.space_dim)]

        def update_lims(coords):
            for i in range(self.space_dim):
                mins[i] = min(mins[i], min(coords[i]))
                maxes[i] = max(maxes[i], max(coords[i]))

        fig, ax = plt.subplots(
            nrows=1, ncols=1, figsize=(7, 7),
            subplot_kw=dict(projection='2d' if self.space_dim == 2 else '3d')
        )

        coords = list(zip(*self.anchors))
        ax.scatter(*coords, marker='o', color='k')
        update_lims(coords)

        if len(self.anchors) == 1: return fig, ax

        for line in itertools.combinations(self.anchors, 2):
            coords = list(zip(*line))
            ax.plot(*coords, marker=None, color='k', linestyle='-')

        if len(self.anchors) == 2: return fig, ax

        for face in itertools.combinations(self.anchors, 3):
            poly_3d = Poly3DCollection([v.get_nparr() for v in face], color='r', alpha=0.2)
            ax.add_collection3d(poly_3d)

        for point in extra_points:
            if isinstance(point, (tuple, list)): point, point_kwargs = point
            else: point_kwargs = {'marker': '^', 's': 40, 'color': 'k'}

            assert isinstance(point, Vector) and point.dim == self.space_dim
            coords = list(zip(*[point]))
            ax.scatter(*coords, **point_kwargs)
            update_lims(coords)

        for line in extra_lines:
            if isinstance(line, (tuple, list)) and len(line) == 2 and isinstance(line[1], dict):
                line, line_kwargs = line
            else:
                line_kwargs = {'marker': '', 'color': 'k'}
                if isinstance(line, (tuple, list)): line = Hypersurface(*line)

            assert isinstance(line, Hypersurface) and line.surface_dim == 1

            coords = list(zip(*line.anchors))
            ax.plot(*coords, **line_kwargs)
            update_lims(coords)

        for face in extra_faces:
            if isinstance(face, (tuple, list)) and len(face) == 2 and isinstance(face[1], dict):
                face, face_kwargs = line
            else:
                face_kwargs = {'color': 'b', 'alpha': 0.2}
                if isinstance(face, (tuple, list)): face= Hypersurface(*face)

            assert isinstance(face, Hypersurface) and face.surface_dim == 2

            poly_3d = Poly3DCollection([v.get_nparr() for v in face.anchors], **face_kwargs)
            ax.add_collection3d(poly_3d)

            coords = list(zip(*face.anchors))
            update_lims(coords)

        D = max(max(mx - mn for mn, mx in zip(mins, maxes)), 0.5)
        eps = min(0.02 * D, 0.25)

        for fn, mn, mx in zip(('set_xlim', 'set_ylim', 'set_zlim'), mins, maxes):
            if hasattr(ax, fn): getattr(ax, fn)(mn-eps, mn+D+eps)

        if view_init_kwargs: ax.view_init(**view_init_kwargs)

        return fig, ax

class SimplicialTiling(Hypersurface):
    def __init__(self, N, vertex_ids=None, dim='match', anchors=None):
        assert isinstance(N, int) and N > 0

        if dim == 'match': dim = N
        assert isinstance(dim, int) and dim >= N

        if anchors is None:
            anchors = np.zeros((N, dim))
            for i in range(N): anchors[i, i] = 1

        self.N = N

        if vertex_ids is None: vertex_ids = tuple(np.arange(N))
        else:
            assert isinstance(vertex_ids, (list, tuple, np.ndarray)), \
                f"vertex_ids must be ordered sequence! Got {type(vertex_ids)}."
            vertex_ids = tuple(vertex_ids)

        assert len(vertex_ids) == N
        self.vertex_ids = vertex_ids

        super().__init__(*anchors)

    def new_reflected(self, new_vertex_ids):
        assert isinstance(new_vertex_ids, (list, tuple, np.ndarray)), \
                f"new_vertex_ids must be ordered sequence! Got {type(new_vertex_ids)}."
        assert len(new_vertex_ids) == self.N

        dropped_vertex_ids = set(self.vertex_ids) - set(new_vertex_ids)
        to_add_vertex_ids = set(new_vertex_ids) - set(self.vertex_ids)
        assert len(dropped_vertex_ids) == 1 and len(to_add_vertex_ids) == 1
        dropped_vertex_id = list(dropped_vertex_ids)[0]
        to_add_vertex_id  = list(to_add_vertex_ids)[0]

        dropped_vertex_idx = self.vertex_ids.index(dropped_vertex_id)

        dropped_anchor = self.anchors[dropped_vertex_idx]
        reflection_face = self.drop(dropped_anchor)

        new_anchors = []
        for i, new_id in enumerate(new_vertex_ids):
            if new_id in self.vertex_ids:
                old_vertex_idx = self.vertex_ids.index(new_id)
                new_anchor = self.anchors[old_vertex_idx]
            else:
                new_anchor = dropped_anchor & reflection_face

            new_anchors.append(new_anchor)

        return SimplicialTiling(self.N, vertex_ids = new_vertex_ids, dim=self.space_dim, anchors=new_anchors)

    def simplicial_to_local(self, x):
        """
        In `simplicial` coordinates, x is a dictionary with values summing to one, with keys self.vertex_ids.
        """
        return self.global_to_local(self.superficial_to_global(x))

    def simplicial_to_global(self, x):
        """
        In `simplicial` coordinates, x is a dictionary with values summing to one, with keys self.vertex_ids.
        """
        assert np.isclose(sum(x.values()), 1), f"sum(x.values) should be 1! Got {sum(x.values())}. x = {x}"
        x_vec = Vector([x[v_id] for v_id in self.vertex_ids])
        anchor_matrix = Matrix(self.anchors).T()
        return anchor_matrix @ x_vec

    def simplicial_to_global_bulk(self, xs):
        """for bulk operations, operates solely within nparrays."""
        for x in xs:
            assert np.isclose(sum(x.values()), 1), f"sum(x.values) != 1! Got {sum(x.values())}. x = {x}"

        anchor_matrix = Matrix(self.anchors).T().get_nparr()
        simplicial_X_arr = np.array([[x[v_id] for v_id in self.vertex_ids] for x in xs]).T

        return (anchor_matrix @ simplicial_X_arr).T

    def global_to_simplicial(self, x):
        np.testing.assert_almost_equal(x.distance_to(self), 0)
        return {v: (x | a).norm() for v, a in zip(self.vertex_ids, self.anchors)}

    def local_to_simplicial(self, x):
        return self.global_to_simplicial(self.local_to_global(x))

    def plot(
        self,
        extra_points=[],
        extra_lines=[],
        extra_faces=[],
        show_vertex_labels=True,
        **view_init_kwargs
    ):
        v_kwargs = dict(elev=45, azim=45) # These defaults are good for a simplex
        v_kwargs = {**v_kwargs, **view_init_kwargs}

        fig, ax = super().plot(
            extra_points=extra_points, extra_lines=extra_lines, extra_faces=extra_faces, **v_kwargs,
        )

        for face in [self] + extra_faces:
            if not isinstance(face, SimplicialTiling): continue

            for anchor, vertex_id in zip(face.anchors, face.vertex_ids):
                ax.text(*anchor, f"{vertex_id}", fontsize=28)

        return fig, ax

def all_paths_of_len(G, n):
    all_paths = set()
    for node in G: all_paths.update(all_paths_from_u_of_len(G, node, n))
    return all_paths

def all_paths_from_u_of_len(G,u,n,excludeSet = None):
    if excludeSet == None: excludeSet = set([u])
    else: excludeSet.add(u)

    if n==0: return frozenset([frozenset([u])])

    paths = frozenset([
        frozenset({u, *path}) for neighbor in G.neighbors(u) if neighbor not in excludeSet \
            for path in all_paths_from_u_of_len(G,neighbor,n-1,excludeSet)
    ])
    excludeSet.remove(u)
    return paths

class LabeledSimplicialManifold():
    def __init__(
        self,
        simplices,
        all_within_4_have_straight_embeds=False,
    ):
        """These 'simplices' are just vertex ids -- not simplicial tilings from above."""
        assert isinstance(simplices, (tuple, list))
        assert len(simplices) > 0

        s0 = simplices[0]
        assert isinstance(s0, (tuple, list, set))
        self.d = len(s0) - 1

        assert self.d == 2, "Only supports 2D manifolds for now."

        # TODO: Check and also set this if there are no 1D boundaries.
        self.all_within_4_have_straight_embeds = all_within_4_have_straight_embeds

        # We'll ensure that any r-NN graph we compute is sufficiently small that we don't need to worry about
        # Going across too many simplices. Using the height of the simplex means that we won't need to worry
        # about anything not sharing a vertex, it turns out.
        # Source for simplex height: https://math.stackexchange.com/questions/1697870/height-of-n-simplex
        self.recommended_max_r = np.sqrt((self.d + 1)/self.d)


        vocab = set(s0)
        for s in simplices:
            assert isinstance(s, (tuple, list, set))
            assert len(s) == len(s0)
            vocab.update(s)

        self.vocab = sorted(list(vocab))
        self.idxmap = {v: i for i, v in enumerate(self.vocab)}

        self.simplices = tuple([frozenset(s) for s in simplices])
        self.simplex_indices = {frozenset(s): i for i, s in enumerate(self.simplices)}

        hyperplane_linkages = {}
        for i, s in enumerate(self.simplices):
            for hyperplane in itertools.combinations(s, self.d):
                hyperplane = frozenset(hyperplane)
                if hyperplane not in hyperplane_linkages: hyperplane_linkages[hyperplane] = []

                hyperplane_linkages[hyperplane].append((s, i))

        assert max(len(v) for v in hyperplane_linkages.values()) <= 2, (
            "A single hyperplane is connected to too many simplices! "
            f"{[(k, v) for k, v in hyperplane_linkages.items() if len(v) > 2]}"
        )

        self.hyperplane_linkages = hyperplane_linkages

        simplex_neighbors = {i: [] for i in range(len(self.simplices))}
        for hyperplane, simplices in self.hyperplane_linkages.items():
            if len(simplices) == 1: continue
            (s1, i1), (s2, i2) = simplices
            simplex_neighbors[i1].append(((s2, i2), hyperplane))
            simplex_neighbors[i2].append(((s1, i1), hyperplane))

        assert min(len(v) for v in simplex_neighbors.values()) >= 1, \
            f"No simplex should be Disconnected! {simplex_neighbors}"

        self.simplex_neighbors = simplex_neighbors

        simplex_nodes = np.arange(len(self.simplices))
        simplex_edges = []
        for simplex_node, neighbors in self.simplex_neighbors.items():
            for (neighbor, neighbor_idx), hyperplane in neighbors:
                simplex_edges.append((simplex_node, neighbor_idx))

        self.simplex_G = nx.Graph()
        self.simplex_G.add_nodes_from(simplex_nodes)
        self.simplex_G.add_edges_from(simplex_edges)

        cutoff = 4 if all_within_4_have_straight_embeds else 3

        # If the path (including x1_simplex and x2_simplex) is too long, its not worth considering.
        # The distance will ultimately end up either being too long or it will be shorter to go through
        # the vertex.
        shortest_simplex_paths_raw = dict(nx.all_pairs_shortest_path(self.simplex_G, cutoff=cutoff))
        all_shortest_simplex_paths = {}
        for v1, v1_shortest_paths in shortest_simplex_paths_raw.items():
            for v2, single_shortest_path in v1_shortest_paths.items():
                assert (v1, v2) not in all_shortest_simplex_paths

                if len(single_shortest_path) > 2:
                    all_shortest_paths = list(nx.all_shortest_paths(self.simplex_G, v1, v2))
                    assert single_shortest_path in all_shortest_paths
                else: all_shortest_paths = [single_shortest_path]

                all_shortest_simplex_paths[(v1, v2)] = all_shortest_paths

        for (v1, v2), all_shortest_paths in all_shortest_simplex_paths.items():
            if (v2, v1) not in all_shortest_simplex_paths:
                all_shortest_simplex_paths[(v2, v1)] = [p[::-1] for p in all_shortest_paths]

        simplicial_tilings = {}
        all_directly_embeddable_simplexes_and_tilings = {}

        for (v1, v2), all_shortest_paths in all_shortest_simplex_paths.items():
            v1_v2_tilings = []
            v1_vertices, v2_vertices = self.simplices[v1], self.simplices[v2]

            for simplex_path in all_shortest_paths:
                S1 = SimplicialTiling(N=(self.d+1), vertex_ids=list(v1_vertices))
                simplices_in_tiling = [S1]

                for simplex_idx in simplex_path[1:]:
                    simplex_vertices = self.simplices[simplex_idx]
                    S = simplices_in_tiling[-1].new_reflected(list(simplex_vertices))
                    simplices_in_tiling.append(S)

                S2 = simplices_in_tiling[-1]
                assert set(S2.vertex_ids) == set(v2_vertices)
                v1_v2_tilings.append(simplices_in_tiling)

                if 1 < len(simplex_path) and len(simplex_path) <= cutoff:
                    simplex_path_key = frozenset(simplex_path)

                    already_captured = False
                    to_pop = []
                    for k in all_directly_embeddable_simplexes_and_tilings:
                        if simplex_path_key.issubset(k):
                            already_captured = True
                        elif simplex_path_key.issuperset(k):
                            # In this case we can drop the (strictly smaller) local patch.
                            to_pop.append(k)

                    for k in to_pop: all_directly_embeddable_simplexes_and_tilings.pop(k)

                    if not already_captured:
                        all_directly_embeddable_simplexes_and_tilings[simplex_path_key] = (
                            simplex_path, simplices_in_tiling
                        )

            simplicial_tilings[(v1, v2)] = v1_v2_tilings

        for v1, v2 in itertools.permutations(simplex_nodes, 2):
            if (v1, v2) not in all_shortest_simplex_paths:
                all_shortest_simplex_paths[(v1, v2)] = []
                simplicial_tilings[(v1, v2)] = []

        self.simplicial_tilings = simplicial_tilings
        self.all_shortest_simplex_paths = all_shortest_simplex_paths
        self.all_directly_embeddable_simplexes_and_tilings = all_directly_embeddable_simplexes_and_tilings

        nodes = self.vocab
        edges = []
        for s in self.simplices: edges.extend(itertools.combinations(s, 2))

        self.G = nx.Graph()
        self.G.add_nodes_from(nodes)
        self.G.add_edges_from(edges)
        self.has_pos = False

        # We use this graph, via graph isomorphism searching, to match topical simplices to manifold
        # simplicial tilings. We use a di
        self.full_G = nx.DiGraph()
        self.full_G.add_nodes_from(self.vocab, type='vertex')
        self.full_G.add_nodes_from(self.simplices, type='simplex')

        # We'll connect vertices together based on bidirectional edges.
        bi_edges = []
        for s in self.simplices: bi_edges.extend(itertools.permutations(s, 2))

        self.full_G.add_edges_from(bi_edges)

        # We'll connect simplex nodes to their vertices with a unidirectional edge from the simplex to the
        # vertex. In this way, there will only ever be *1* node that connects with unidirectional edges to the
        # vertices of any simplex. this will help us ensure any graph isomorphism also implies a simplicial
        # alignment.
        self.full_G.add_edges_from((simplex, v) for simplex in self.simplices for v in simplex)

    def _compute_pos(self):
        self._simplex_G_pos = graphviz_layout(self.simplex_G, prog="neato")
        self._full_G_pos = graphviz_layout(self.full_G, prog="neato")

        try:
            self._pos           = nx.planar_layout(self.G)
        except nx.NetworkXException as e:
            print(f"No planar embedding exists: {e}\nTrying neato")
            self._pos           = graphviz_layout(self.G, prog="neato")

        self.has_pos = True

    def display(
        self, figsize=5, do_print=True, axes=None, labels_to_display='ALL'
    ):
        if not self.has_pos: self._compute_pos()

        if do_print:
            print(f"Manifold has {len(self.simplices)} simplices")

        if axes is None:
            fig, (ax_full, ax_simplex) = plt.subplots(nrows=1, ncols=2, figsize=(2*figsize, figsize))
        else:
            fig = None
            ax_full, ax_simplex = axes

        draw_kwargs = {'vmin': 0.0, 'vmax': 1.0, 'node_color': 'k'}

        ax_full.set_title("Simplicial Complex (all vertices)")
        nx.draw(
            G=self.G, pos=self._pos, with_labels=True, ax=ax_full, font_color='white', node_size=240,
            **draw_kwargs
        )

        ax_simplex.set_title("Simplicial Complex (node per simplex)")
        nx.draw(
            G=self.simplex_G, pos=self._simplex_G_pos, with_labels=False, ax=ax_simplex, node_size=40,
            **draw_kwargs
        )
        nx.draw_networkx_labels(
            G=self.simplex_G, ax=ax_simplex,
            pos={k: (x+35, y) for k, (x, y) in self._simplex_G_pos.items()},
            labels={i: f"({'-'.join(str(v) for v in self.simplices[i])})" for i in self.simplex_G}
        )
        mn, mx = ax_simplex.get_xlim()
        ax_simplex.set_xlim(mn-5, mx+80)

        return fig, (ax_full, ax_simplex)

    def efficient_pairwise_distances(self, xs):
        # Let's check validity (& grab their associated simplices/unified key order at the same time)
        simplices = []
        simplex_key_orders = {}
        x_idxs_by_simplex = {}
        for x_idx, x in enumerate(xs):
            assert isinstance(x, dict) and len(x) == (self.d + 1) and all(v >= 0 for v in x.values()), \
                f"{x} is invalid! {self.d}"
            np.testing.assert_almost_equal(sum(x.values()), 1)

            vs = frozenset(x.keys())
            assert vs in self.simplex_indices

            simplex_idx = self.simplex_indices[vs]
            simplices.append(simplex_idx)
            simplex_key_orders[simplex_idx] = tuple(sorted(vs))

            if simplex_idx not in x_idxs_by_simplex: x_idxs_by_simplex[simplex_idx] = [x_idx]
            else: x_idxs_by_simplex[simplex_idx].append(x_idx)

        # We'll store the xs in a frozen list for efficiency.
        xs_keys = tuple([simplex_key_orders[s_idx] for x, s_idx in zip(xs, simplices)])
        xs_vals = tuple([tuple([x[k] for k in ks]) for x, ks in zip(xs, xs_keys)])

        N = len(xs)
        x_idxs = np.arange(N)
        def x_idx_map(x):
            x_k = tuple(sorted(x.keys()))
            x_v = tuple([x[k] for k in x_k])
            for i, (k, v) in enumerate(zip(xs_keys, xs_vals)):
                if x_k == k and x_v == v: return i

            raise KeyError(f"Can't find {x} in {xs}!")

        # Now we can compute distances in larger swaths.
        # Let's initialize our distance matrix, which we'll use as our container.
        distance_matrix = np.ones((N, N)) * float('inf')
        for i in range(N): distance_matrix[i, i] = 0

        for _, (simplices, tiling) in self.all_directly_embeddable_simplexes_and_tilings.items():
            all_X_global = []
            all_X_idxs = []
            for simplex_idx, simplex_surface in zip(simplices, tiling):
                if simplex_idx not in x_idxs_by_simplex:
                    print(f"No points ovserved on simplex {simplex_idx}")
                    continue

                x_idxs = x_idxs_by_simplex[simplex_idx]
                all_X_idxs.extend(x_idxs)

                simplex_X_global = simplex_surface.simplicial_to_global_bulk([xs[x_idx] for x_idx in x_idxs])
                all_X_global.append(simplex_X_global)

            all_X_global = np.concatenate(all_X_global, axis=0)
            simplex_tiling_N = len(all_X_idxs)

            pairwise_distances = pdist(all_X_global)

            for i in range(simplex_tiling_N):
                for j in range(i+1, simplex_tiling_N):
                    condensed_idx = simplex_tiling_N * i + j - ((i + 2) * (i + 1)) // 2

                    x_idx_i, x_idx_j = all_X_idxs[i], all_X_idxs[j]
                    if pairwise_distances[condensed_idx] < distance_matrix[x_idx_i][x_idx_j]:
                        distance_matrix[x_idx_i][x_idx_j] = pairwise_distances[condensed_idx]
                        distance_matrix[x_idx_j][x_idx_i] = pairwise_distances[condensed_idx]

        if self.all_within_4_have_straight_embeds: return distance_matrix

        for i in range(N):
            for j in range(i+1, N):
                if np.isinf(distance_matrix[i][j]):
                    # In this case, we found no way to directly embed both of these in a convex patch, so
                    # we'll fall back on the slower, approximate_geodesic method.
                    dist = self.approximate_geodesic_distance(xs[i], xs[j])
                    distance_matrix[i][j] = dist
                    distance_matrix[j][i] = dist

        return distance_matrix

    def approximate_geodesic_distance(self, x1, x2):
        # First, some housekeeping

        simplices, vertices = [], []
        for x in (x1, x2):
            assert isinstance(x, dict) and len(x) == (self.d + 1) and all(v >= 0 for v in x.values())
            np.testing.assert_almost_equal(sum(x1.values()), 1)

            vs = frozenset(x.keys())
            vertices.append(vs)
            simplices.append(self.simplex_indices[vs])

        x1_simplex, x2_simplex = simplices
        x1_vertices, x2_vertices = vertices

        # There are a few relevant cases here. Firstly, x1 & x2 can both live on a single simplex.
        # In that case, computing distance is quite trivial.
        if x1_simplex == x2_simplex:
            S = SimplicialTiling(N=len(x1_vertices), vertex_ids=list(x1_vertices))
            return S.simplicial_to_global(x1).distance_to(S.simplicial_to_global(x2))

        # In the case where they don't share a simplex, things aren't quite so simple. We're going to impose
        # a cutoff, based on how many simplices separate the point --- if its too many, the answer is
        # float('inf'). As our only point is to use this for an rNN graph, that'll be fine
        # (provided r is small). But we need to handle a few of the more local cases.

        through_vertex_distance, straight_line_distance = float('inf'), float('inf')

        # One case worth analyzing is when they share a vertex. In this case, no matter how far apart
        # things are in terms of separating simplices, we can always shortcut through the vertex
        # (in a manner that would be hard to capture once things are projected to R^2).
        shared_vertices = x1_vertices.intersection(x2_vertices)
        if shared_vertices:

            # Note we require that these be planar simplicies here. Otherwise, we don't just need to consider
            # paths through single vertices, but rather paths through shared edges that can't be represented
            # by the straight-line distance in a simplicial tiling, for example. Those are trickier than
            # just a vertex as we need to determine *where* on that shared edge they intersect.
            assert self.d == 2

            raw_S1 = SimplicialTiling(N=len(x1_vertices), vertex_ids=list(x1_vertices))
            x1_on_S1 = raw_S1.simplicial_to_global(x1)

            raw_S2 = SimplicialTiling(N=len(x2_vertices), vertex_ids=list(x2_vertices))
            x2_on_S2 = raw_S2.simplicial_to_global(x2)

            for v in shared_vertices:
                x1_v = {x1v: 1 if x1v == v else 0 for x1v in x1_vertices}
                x2_v = {x2v: 1 if x2v == v else 0 for x2v in x2_vertices}

                x1_to_v = x1_on_S1.distance_to(raw_S1.simplicial_to_global(x1_v))
                x2_to_v = x2_on_S2.distance_to(raw_S2.simplicial_to_global(x2_v))

                through_vertex_distance = min(through_vertex_distance, x1_to_v + x2_to_v)

        # Next, we'll consider the distance going through a small # of simplices.
        try:
            shortest_simplex_paths = self.all_shortest_simplex_paths[(x1_simplex, x2_simplex)]
            simplicial_tilings     = self.simplicial_tilings[(x1_simplex, x2_simplex)]
        except KeyError as e:
            print(f"Failed to find shortest path from {x1_simplex} to {x2_simplex}: {e}")
            print(f"Has keys: {self.all_shortest_simplex_paths.keys()}")
            print("Trying the reverse")
            shortest_simplex_paths = self.all_shortest_simplex_paths[(x2_simplex, x1_simplex)][::-1]
            simplicial_tilings     = self.simplicial_tilings[(x2_simplex, x1_simplex)][::-1]

        for simplex_path, simplices_in_tiling in zip(shortest_simplex_paths, simplicial_tilings):
            # Note that this 4 is dependent on this being a 2d manifold! Otherwise this wouldn't be valid.
            assert self.d == 2
            if len(simplex_path) > 4:
                print(
                    f"Warning: Pre-computed simplex path from {x1_simplex} to {x2_simplex} has len "
                    f"{len(simplex_path)}: {simplex_path}\n"
                    "Lengths are typically capped at <= 4 as anything separated by more than 2 simplices is"
                    "not handleable yet. Skipping path."
                )
                continue

            # Now that we have a sufficiently short path
            # (which, it turns out, not unintentionally, we can embed directly)
            # we need to build a single simplicial tiling with that path so we can then compute the
            # distance between x1 and x2 in that tiling.

            S1 = simplices_in_tiling[0]
            S2 = simplices_in_tiling[-1]

            x1_in_ST = S1.simplicial_to_global(x1)
            x2_in_ST = S2.simplicial_to_global(x2)

            # Now, there are two cases we need to consider -- either we can directly draw a line from
            # x1 to x2 _and stay in the simplicial tiling_, or we can't, in which case the distance
            # will be shorter going through the vertex, which we've already covered above.
            #
            # To check if it leaves the simplex, we'll find all boundary edges and check if the line from
            # x1_in_ST to x2_in_ST crosses any of them.

            # First, let's form the line from x1 to x2.
            x1_to_x2 = Hypersurface(x2_in_ST, x1_in_ST)

            # TODO: this part also relies on things being planar.
            assert self.d == 2
            edges_to_simplices = {}
            for s in simplices_in_tiling:
                for e in itertools.combinations(s.vertex_ids, self.d):
                    e = frozenset(e)
                    if e in edges_to_simplices: edges_to_simplices[e].append(s)
                    else: edges_to_simplices[e] = [s]

            crosses_boundary = False
            any_line_parallel_missing_simplex_edge = False
            boundary_edges = {e: s_list[0] for e, s_list in edges_to_simplices.items() if len(s_list) == 1}

            boundary_edges_lookup = []
            for (v1, v2), boundary_simplex in boundary_edges.items():
                v1_coords = {v: 1 if v == v1 else 0 for v in boundary_simplex.vertex_ids}
                v2_coords = {v: 1 if v == v2 else 0 for v in boundary_simplex.vertex_ids}

                v1_in_ST = boundary_simplex.simplicial_to_global(v1_coords)
                v2_in_ST = boundary_simplex.simplicial_to_global(v2_coords)

                boundary_edges_lookup.append(frozenset((v1_in_ST.freeze(), v2_in_ST.freeze())))
            boundary_edges_lookup = set(boundary_edges_lookup)

            for edge, boundary_simplex in boundary_edges.items():
                v1, v2 = edge
                v1_coords = {v: 1 if v == v1 else 0 for v in boundary_simplex.vertex_ids}
                v2_coords = {v: 1 if v == v2 else 0 for v in boundary_simplex.vertex_ids}

                v1_in_ST = boundary_simplex.simplicial_to_global(v1_coords)
                v2_in_ST = boundary_simplex.simplicial_to_global(v2_coords)

                edge_L = Hypersurface(v1_in_ST, v2_in_ST)

                try:
                    (
                        is_parallel, is_collinear,
                        (endpoints_coincident, endpoints_coincident_interior, endpoints_coincident_endpoints),
                        interiors_intersect
                    ) = edge_L.segments_intersect(x1_to_x2)
                except AssertionError as e:
                    chained_err = (
                        "Failed to compute boundary intersection!\n"
                        f"  simplex path = {simplex_path}\n"
                        f"  v1_coords = {v1_coords}, v2_coords = {v2_coords}\n"
                        f"  x1_coords = {x1}, x2_coords = {x2}\n"
                        f"Original error (in (v2 - v1).segments_intersect(x2 - x1)):\n"
                        f"{e}\n"
                        f"Traceback:\n"
                        f"{traceback.print_exc()}"
                    )
                    raise AssertionError(chained_err)

                if is_parallel:
                    # If they're parallel, it doesn't cross the boundary for sure.
                    continue
                elif interiors_intersect:
                    # If they *aren't* parallel and their interiors intersect, then it definitely crosses.
                    crosses_boundary = True
                    break
                elif endpoints_coincident_interior or endpoints_coincident_endpoints:
                    # This case is a little funny -- baically, the line could have a point at an endpoint and
                    # either go immediately out of the simplex, and potentially end on another boundary edge
                    # (which would thus also map to this case), while being totally out of the simplex. E.g.,
                    #    *----O
                    #   / \  /|
                    #  /   \/ :
                    #  *----* |
                    #  \   /\ ;
                    #   \ /  \|
                    #    *----O
                    # Where the solid lines are simplex edges, the "*"s are simplex vertices, the "O"s are our
                    # line x1_to_x2 endpoints, and the dash-dot line is the line x1_to_x2.
                    #
                    # In this case, we definitely want this to flag as a "crossing", but our cases above won't
                    # catch it. To do so, we need to compare our line to a line from the same endpoint that we
                    # know is in the simplex.

                    # First let's find the coincident endpoint
                    # TODO: this is duplicate work!
                    x1_on_edge = x1_in_ST | edge_L
                    x2_on_edge = x2_in_ST | edge_L

                    x1_to_edge = x1_on_edge - x1_in_ST
                    x2_to_edge = x2_on_edge - x2_in_ST

                    if np.isclose(x1_to_edge.norm(), 0):
                        coincident_endpoint, other_endpoint = x1_in_ST, x2_in_ST
                    elif np.isclose(x1_to_edge.norm(), 0):
                        coincident_endpoint, other_endpoint = x2_in_ST, x1_in_ST
                    else:
                        # In this case, this boundary's endpoint intersects with a cointuation of the line
                        # x1_to_x2 -- that doesn't concern us, as it may be totally fine.
                        continue

                    # Now we want to know if the line leaving the coincident endpoint is heading into the
                    # simplex tiling or out of the simplex tiling. This is, surprisingly, tricky to do
                    # exactly. There are just a ton of cases here that could come up, where things are heading
                    # almost any direction off the boundary and still be in the ST, because there could be
                    # other simplexes there the "boundary edge" doesn't know about. But, we have the advantage
                    # that we're looking across all boundary edges at once. So, here, we'll check if the line
                    # is traveling strictly inside the reflection of a simplex across the boundary edge ---
                    # e.g., the missing simplex.
                    # Annoyingly, there is still an edge case to this -- when the line leaving is parallel to
                    # the border of this boundary simplex. It's difficult (maybe impossible) to know without
                    # looking at other simplices whether this is traveling along the boundary to another
                    # simplex in the tiling, which is legal, or heading out into freespace along an
                    # unconnected line, which is not. E.g., differentiating the below
                    #           ()----()                  ()----()
                    #          /  \  /  \                /  \  /  \
                    #         /    \/    \              /    \/    \
                    #        ()----()----@@            ()----()----@@
                    #       /  \  /      :      vs    /  \  /  \  /
                    #      /    \/     :             /    \/    \/
                    #     ()----()----@@----()      ()----()----@@----()
                    #       \  /  \  /  \  /          \  /  \  /  \  /
                    #        \/    \/    \/            \/    \/    \/
                    #        ()----()----()            ()----()----()
                    #
                    # Where the simplex edges are given by ---- or / or \ and the endpoints of the line in
                    # question are given by "@@" signs. On the left, there is no simplex boundary spanning the
                    # "@@" points, and the implicit line would be parallel to any "missing" simplex edge, and
                    # thus checking whether the line is in the interior of the simplex implied by the boundary
                    # edge won't catch it. However, we can't just immediately decide that any such case should
                    # be flagged, as the case on the right looks perfectly identical from the perspective of
                    # the boundary edge in the lower right (that's horizontal), but now there actually is a
                    # simplex filling in that hole and the line is valid.
                    #
                    # To catch this case, we'll check if the edge we're parallel to in the missing simplex is
                    # a different boundary edge in the simplex. If it is, then that is fine, if it isn't, then
                    # it is a problem. Note that there is, somehow, *still* a possibility that maybe the
                    # "missing" simplex that we're parallel to is just not included in the path, but actually
                    # is in the manifold, but that... I don't know how to handle that case.  Given how short
                    # are paths are, I think this is actually fine but still, it could be an issue.

                    new_fake_vertex = -1
                    assert new_fake_vertex not in boundary_simplex.vertex_ids
                    missing_simplex = boundary_simplex.new_reflected([v1, v2, new_fake_vertex])

                    new_vertex_in_ST = missing_simplex.simplicial_to_global({
                        v1: 0, v2: 0, new_fake_vertex: 1,
                    })
                    missing_simplex_center = missing_simplex.simplicial_to_global({
                        v1: 1/3, v2: 1/3, new_fake_vertex: 1/3
                    })


                    # Here we need to check if its going into the missing simplex or outside it even if not
                    # going straight back internally. We'll iteratively flag its position relative to all of
                    # the missing simplex's edges in a loop to keep the code DRY(ish).
                    missing_simplex_edge1 = edge_L
                    missing_simplex_edge2 = Hypersurface(v1_in_ST, new_vertex_in_ST)
                    missing_simplex_edge3 = Hypersurface(v2_in_ST, new_vertex_in_ST)

                    # Contents: [(shared_vertex, is_collinear, edge_is_boundary, ends_outside)... ]
                    direction_properties = []

                    for i, missing_simplex_edge in enumerate(
                        (missing_simplex_edge1, missing_simplex_edge2, missing_simplex_edge3)
                    ):
                        lookup_edge = frozenset((a.freeze() for a in missing_simplex_edge.anchors))
                        edge_is_boundary = lookup_edge in boundary_edges_lookup
                        if i == 0:
                            # Just to check, this is actually the known boundary edge...
                            assert edge_is_boundary, f"Known boundary edge not flagged!"

                        edge_to_other_endpoint = (
                            other_endpoint - (other_endpoint | missing_simplex_edge)
                        )
                        other_endpoint_on_edge = np.isclose(edge_to_other_endpoint.norm(), 0)

                        edge_to_coincident_endpoint = (
                            coincident_endpoint - (coincident_endpoint | missing_simplex_edge)
                        )
                        coincident_endpoint_on_edge = np.isclose(edge_to_coincident_endpoint.norm(), 0)

                        shared_vertex = coincident_endpoint_on_edge
                        is_collinear = other_endpoint_on_edge and coincident_endpoint_on_edge
                        if other_endpoint_on_edge:
                            ends_outside = False
                        else:
                            edge_to_missing_simplex_center = (
                                missing_simplex_center - (missing_simplex_center | missing_simplex_edge)
                            )
                            cos_between = Vector.cos_between(
                                edge_to_missing_simplex_center, edge_to_other_endpoint
                            )

                            non_planar_error_message = (
                                "Segments appear to be non-planar?\n"
                                f"  missing_simplex_edge           = {str(missing_simplex_edge)}\n"
                                f"  other_endpoint                 = {str(other_endpoint)}\n"
                                f"  missing_simplex_center         = {str(missing_simplex_center)}\n"
                                f"  edge_to_other_endpoint         = {str(edge_to_other_endpoint)}\n"
                                 "  edge_to_missing_simplex_center = "
                                                           f"{str(edge_to_missing_simplex_center)}\n"
                            )
                            np.testing.assert_almost_equal(
                                np.abs(cos_between), 1, err_msg = non_planar_error_message
                            )

                            ends_outside = np.isclose(cos_between, -1)

                        direction_properties.append(
                            (shared_vertex, is_collinear, edge_is_boundary, ends_outside)
                        )


                    # Now that we've analyzed all edges, we can make final determinations. We have two rules:
                    #
                    # First, if any edge shares a vertex and the other endpoint ends outside the simplex, then
                    # it *can't* enter the missing simplex.
                    #
                    # Second, if any edge of the missing simplex that is also a boundary edge to the
                    # simplicial tiling overall is collinear with our line, then its running alongside a
                    # different boundary edge so does not enter a missing simplex.
                    #
                    # If neither of these are ever triggered, then the new line does enter the missing
                    # simplex, guaranteeably.

                    line_entering_missing_simplex = True
                    for (shared_vertex, is_collinear, edge_is_boundary, ends_outside) in direction_properties:
                        if (shared_vertex and ends_outside) or (is_collinear and edge_is_boundary):
                            line_entering_missing_simplex = False
                            break

                    if line_entering_missing_simplex:
                        crosses_boundary = True
                        break

            if crosses_boundary: continue

            # If we never cross the boundary, then let's get the straight line distance and keep looking
            # through other shortest paths!
            straight_line_distance = min(straight_line_distance, Vector.dist(x1_in_ST, x2_in_ST))

        # Now we've gone through both all possible straight line distances and through any shared vertices.
        # So, we can simply return the minimum!

        return min(straight_line_distance, through_vertex_distance)

    def radius_nearest_neighbor_graph(self, r, *xs, precomputed_distance_matrix=None):
        if r > self.recommended_max_r:
            print(f"Warning! r ({r}) greater than recommended value ({self.recommended_max_r})")

        N = len(xs)
        if precomputed_distance_matrix is None:
            distance_matrix = self.efficient_pairwise_distances(xs)
        else:
            assert precomputed_distance_matrix.shape == (N, N)
            distance_matrix = precomputed_distance_matrix

        nodes = np.arange(N)
        G = nx.Graph()
        G.add_nodes_from(nodes)

        for i in nodes:
            neighbors,  = np.where(distance_matrix[i] < r)
            G.add_edges_from((i, j) for j in neighbors)

        return G, distance_matrix

    def sample_points(self, pts_per_simplex):
        out_pts = []
        for simplex in self.simplices:
            for i in range(pts_per_simplex):
                xs = np.random.rand(self.d+1)
                xs /= xs.sum()
                out_pts.append({v: x for v, x in zip(simplex, xs)})
        return out_pts
