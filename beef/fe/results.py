class Results:
    def __init__(self, analysis, element_results=['M', 'N', 'V'], node_results=[]):
        self.analysis = copy(analysis)
        self.output = None
        self.element_results = element_results
        self.node_results = node_results
        self.element_cogs = np.array([el.get_cog() for el in self.analysis.part.elements])
        
    def process(self, print_progress=True, nonlinear=True):
        self.output = dict()
        for key in self.element_results:
            self.output[key] = np.zeros([len(self.analysis.part.elements), len(self.analysis.t)])
        
        for key in self.node_results:
            self.output[key] = np.zeros([len(self.analysis.part.nodes), len(self.analysis.t)])
            
        # Initiate progress bar
        if print_progress:
            progress_bar = tqdm(total=len(self.analysis.t)-1, initial=0, desc='Post processing')        

        for k, ti in enumerate(self.analysis.t):
            if nonlinear:
                self.analysis.part.deform_part(self.analysis.u[:, k], update_tangents=False)
            else:
                self.analysis.part.deform_part_linear(self.analysis.u[:, k])

            for out in list(self.element_results):
                self.output[out][:, k] = np.array([el.extract_load_effect(out) for el in self.analysis.part.elements])
                
    # CORE METHODS
    def __str__(self):
        return f'BEEF Results ({len(self.steps)} steps, {self.assembly} assembly)'

    def __repr__(self):
        return f'BEEF Results ({len(self.steps)} steps, {self.assembly} assembly)'

