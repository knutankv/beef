'''
FE objects submodule: force definitions
'''

from scipy.interpolate import interp1d
import numpy as np

class Force:
    '''
    Force definition class. 

    Arguments
    -------------
    node_labels : int
        list of node labels to excite
    dofs : int
        list/array of local DOFs to excite
    amplitudes : float
        array defining amplitudes; shape should be either n_nodelabels x n_samples (length of time axis) or n_samples x 1 in dimensions - if latter
        the same force is assumed for all node labels
    name : 'Force-0', optional
        name of force object
    plotcolor : 'DarkOrange', optional
        color used for plotting of analysis with force definitions
    t : None, optional
        time axis corresponding to input force history (n_samples)

    Notes 
    ------
    Generates interpolant object that can be evaluated at time instance to generate force vector in global format, i.e., `self.evaluate`

    '''

    def __init__(self, node_labels, dofs, amplitudes, name='Force-0', plotcolor='DarkOrange', t=None):
        self.plotcolor = plotcolor
        self.name = name
        self.dof_ix = self.adjust_dof_ix(dofs, len(node_labels))
        amplitudes = self.adjust_amplitudes(amplitudes, len(node_labels))

        self.min_dt = np.inf
        self.t = t
        self.node_labels = node_labels

        if t is None:
            self.evaluate = lambda __: amplitudes[:, 0]  # output constant force regardless
        else:
            if np.array(t).ndim==0:
                t = np.array([0, t])    # assume max time is specified (linear ramp)
                if amplitudes.shape[1] == 1:
                    amplitudes = np.hstack([amplitudes*0, amplitudes])
            else:
                self.min_dt = np.min(np.diff(t))

            if amplitudes.shape != tuple([len(node_labels), len(t)]):
                raise ValueError('Please fix form of amplitude input. It should be either n_nodelabels x n_samples or n_samples x 1 in dimensions.')

            self.evaluate = interp1d(t, amplitudes, fill_value=amplitudes[:, 0]*0, bounds_error=False)
            self.amplitudes = amplitudes

    # CORE METHODS
    def __str__(self):
        return f'BEEF Force: {self.name}'

    def __repr__(self):
        return f'BEEF Force: {self.name}'

    @staticmethod
    def adjust_dof_ix(dix, n_nodes):
        '''
        Adjust DOF index format.
        '''
        if np.array(dix).ndim != 2:
            if np.array(dix).ndim == 0:
                dix = [[dix]]*n_nodes
            elif np.array(dix).ndim == 1:
                dix = [dix]*n_nodes

        return dix

    @staticmethod
    def adjust_amplitudes(amplitudes, n_nodes):
        '''
        Adjust amplitudes array to match the number of nodes.

        Arguments
        -------------
        amplitudes : float 
            array defining amplitudes; shape should be either n_nodelabels x n_samples (length of time axis) or 
            n_samples x 1 in dimensions - if latter the same force is assumed for all node labels
        n_nodes : int
            number of nodes

        Returns
        -------------
        amplitudes : float
            adjusted array defining amplitudes, corrected to match the required format to generate the force interpolant callable with `Force.evaluate`.

        '''

        if type(amplitudes) is list:
            amplitudes = np.array(amplitudes)
        elif type(amplitudes) in [float, int]:
            amplitudes = np.array([amplitudes])
        
        if amplitudes.ndim == 1:
            amplitudes = amplitudes[np.newaxis, :]

        if amplitudes.shape[0] == 1 and n_nodes>1:
            amplitudes = np.repeat(amplitudes, n_nodes, axis=0)

        return amplitudes

def get_element_equivalent_loads(elements, fun=None, mass_per_length=None, total_mass=None, g=9.81,
                                 direction=None, include_moments=True):
    r'''
    Consistent equivalent (fixed-end) load vectors of a constant distributed load, per element, in the element's local
    coordinate system.

    The arguments define the load exactly as in `create_nodal_load_from_element_load`. The vectors are used both to
    establish the nodal loads and to correct the element section forces in post-processing: for an element with a
    distributed load, the end forces are \(\{q\} = [k]\{u\} - \{f_{eq}\}\) (local DOFs).

    Arguments
    ------------
    elements : obj
        list of `BeamElement2d`/`BeamElement3d` objects
    fun, mass_per_length, total_mass, g, direction, include_moments
        see `create_nodal_load_from_element_load`

    Returns
    ------------
    f_eq : dict
        element label -> numpy array with the local equivalent load vector (6 values in 2D: u1, v1, theta1, u2, v2,
        theta2; 12 values in 3D: u, v, w, theta_x, theta_y, theta_z at node 1 and node 2)
    '''

    elements = list(elements)
    if len(elements) == 0:
        return {}

    if sum(x is not None for x in [fun, mass_per_length, total_mass]) > 1:
        raise ValueError('Specify at most one of fun, mass_per_length and total_mass.')

    domain = elements[0].domain
    if any(el.domain != domain for el in elements):
        raise ValueError('All elements must be in the same domain (2d or 3d).')

    dim = 2 if domain == '2d' else 3

    if direction is None:
        direction = [0, -1] if dim == 2 else [0, 0, -1]
    direction = np.asarray(direction, dtype=float).flatten()[:dim]
    if direction.size != dim or np.linalg.norm(direction) == 0:
        raise ValueError(f'direction must be a non-zero vector with {dim} components.')
    direction = direction/np.linalg.norm(direction)

    lengths = np.array([el.get_length(undeformed=True) for el in elements])

    # Load intensity (force per unit length) of each element
    if fun is not None:
        q = np.array([float(fun(el.section)) for el in elements])
    elif mass_per_length is not None:
        q = np.ones(len(elements))*float(mass_per_length)*g
    elif total_mass is not None:
        q = np.ones(len(elements))*float(total_mass)/np.sum(lengths)*g
    else:
        q = np.array([float(el.section.get_m()) for el in elements])*g

    f_eq = {}
    for el, qi, L in zip(elements, q, lengths):
        T = np.asarray(el.tmat, dtype=float)
        q_loc = T[:dim, :dim] @ (qi*direction)      # load per unit length in local csys

        if dim == 2:    # local DOFs: u1, v1, theta1, u2, v2, theta2
            f_loc = np.array([q_loc[0]*L/2, q_loc[1]*L/2, q_loc[1]*L**2/12,
                              q_loc[0]*L/2, q_loc[1]*L/2, -q_loc[1]*L**2/12])
            if not include_moments:
                f_loc[[2, 5]] = 0.0
        else:           # local DOFs: u, v, w, theta_x, theta_y, theta_z (x2); theta_z = dv/dx, theta_y = -dw/dx
            qx, qy, qz = q_loc
            f_loc = np.zeros(12)
            f_loc[[0, 6]] = qx*L/2
            f_loc[[1, 7]] = qy*L/2
            f_loc[[2, 8]] = qz*L/2
            if include_moments:
                f_loc[5], f_loc[11] = qy*L**2/12, -qy*L**2/12
                f_loc[4], f_loc[10] = -qz*L**2/12, qz*L**2/12

        f_eq[el.label] = f_eq.get(el.label, 0) + f_loc

    return f_eq


def create_nodal_load_from_element_load(elements, fun=None, mass_per_length=None, total_mass=None, g=9.81,
                                        direction=None, include_moments=True, t=None, name='Element load'):
    r'''
    Establish equivalent nodal loads (`Force` objects) from a constant distributed load on beam elements.

    The load intensity (force per unit length) of each element is defined by one of the following options
    (give at most one of `fun`, `mass_per_length` and `total_mass`):

    * `fun`: function of the section of the element, returning the load per unit length,
      e.g. `lambda sec: sec.rho*sec.A*9.81`
    * `mass_per_length`: constant mass per unit length \(m\), giving \(q = m g\)
    * `total_mass`: total mass \(M\) distributed uniformly over the total length of the elements, giving
      \(q = M g / \sum L\)
    * none of the above (default, gravity): the mass per unit length of the section of each element
      (`section.m`, which equals \(\rho A\) unless `m` is given explicitly for the section), giving \(q = m g\)

    The load acts in the global direction `direction` (per unit length of the element). The consistent equivalent
    nodal loads of a uniform load are used: \(qL/2\) at each node and, from the transverse part, the fixed-end
    moments \(\pm qL^2/12\). They are established in the local coordinate system of each element and
    transformed to the global system, and contributions from elements sharing a node are summed.

    Arguments
    ------------
    elements : obj
        list of `BeamElement2d`/`BeamElement3d` objects (e.g. `part.elements` or a subset of them)
    fun : function, optional
        function of the `Section` object of an element returning the load per unit length
    mass_per_length : float, optional
        constant mass per unit length (multiplied by `g`)
    total_mass : float, optional
        total mass distributed uniformly (per unit length) over all elements (multiplied by `g`)
    g : 9.81, optional
        acceleration of gravity, used with `mass_per_length`, `total_mass` and the default gravity option
    direction : float, optional
        global direction of the load (normalized internally); defaults to gravity, i.e., [0, 0, -1] in 3D and
        [0, -1] in 2D
    include_moments : True, optional
        whether or not to include the consistent nodal moments (if False, the load is lumped as forces only)
    t : None, optional
        passed to `Force`: None gives constant loads, a scalar gives a linear ramp from 0 at t=0 to the full load at t
    name : 'Element load', optional
        base name of the `Force` objects

    Returns
    ------------
    forces : obj
        list of `Force` objects (one per loaded DOF index, each including all nodes with non-zero load in that DOF),
        to be passed to `Analysis` together with other forces

    Example
    ------------
    Self weight of all elements of a part, and a superimposed constant line load on a subset of elements:

        forces = fe.create_nodal_load_from_element_load(part.elements)
        forces += fe.create_nodal_load_from_element_load(part.get_elements([1, 2, 3]), mass_per_length=500.0)
        analysis = fe.Analysis(part, forces=forces)

    Notes
    ------------
    beef recovers element section forces from the element stiffness and the nodal displacements only. For elements
    carrying a distributed load, the correct end forces are \(k u - f_{eq}\), where \(f_{eq}\) are the element's
    equivalent (fixed-end) loads; these are available from `get_element_equivalent_loads` (same arguments).
    '''

    elements = list(elements)
    if len(elements) == 0:
        return []

    f_eq = get_element_equivalent_loads(elements, fun=fun, mass_per_length=mass_per_length, total_mass=total_mass,
                                        g=g, direction=direction, include_moments=include_moments)
    n_dofs = 3 if elements[0].domain == '2d' else 6

    nodal = {}      # node label -> load vector (all DOFs of node)
    for el in elements:
        f_glob = np.asarray(el.tmat, dtype=float).T @ f_eq[el.label]
        for k, node in enumerate(el.nodes):
            nodal.setdefault(node.label, np.zeros(n_dofs))
            nodal[node.label] += f_glob[k*n_dofs:(k+1)*n_dofs]

    labels = list(nodal.keys())
    values = np.array([nodal[lbl] for lbl in labels])
    tol = 1e-12*max(np.max(np.abs(values)), 1e-300)

    forces = []
    for dof in range(n_dofs):
        nonzero = np.abs(values[:, dof]) > tol
        if np.any(nonzero):
            node_labels = [labels[i] for i in np.where(nonzero)[0]]
            amplitudes = values[nonzero, dof][:, np.newaxis]
            forces.append(Force(node_labels, dof, amplitudes, name=f'{name} (dof {dof})', t=t))

    return forces
