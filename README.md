# Google DeepMind GraphCast and GenCast

##########################
These version has been edited by Kirsten Tempest.

## Aim of version 3 (archive_3): ##
- Extract latent features from each MLP by storing them in the state. 
- Save original and updated latent features within Graphcast call. 

# Main changes for version 3: 

- Added in graphcast.py:

def save_latents(x):
    x_np = np.array(x).astype(np.float32)  # create np array from jax array 
    print("Saving latent_mesh_nodes with dtype ", x_np.dtype)
    print("First cell: ", x_np[0][0][0])
    np.save("latent_mesh_nodes.npy", x_np)
    print("Saved latent_mesh_nodes")

def save_updated_latents(x):
    x_np = np.array(x).astype(np.float32)
    print("Saving updated_latent_mesh_nodes with dtype", x_np.dtype)
    print("First cell: ", x_np[0][0][0])
    np.save("updated_latent_mesh_nodes.npy", x_np)
    print("Saved updated_latent_mesh_nodes")
    
 Added within GraphCast __call__ method: 
   
 jax.debug.callback(save_latents, latent_mesh_nodes)
 jax.debug.callback(save_updated_latents, updated_latent_mesh_nodes)
   
 updated_latent_mesh_nodes, latent_graphs_m = self._run_mesh_gnn(latent_mesh_nodes)
 hk.set_state("latent_graphs_m", latent_graphs_m)

##########################

 - Added in deep_typed_graph_net.py: 
 
 Added within DeepTypedGraphNet __call__ method: 
 
 latent_graphs_m = self._process(latent_graph_0, processor_networks)

 # Compute outputs from the last latent graph (if applicable).
 return self._output(latent_graphs_m[-1], decoder_network), latent_graphs_m
 
Updated DeepTypedGraphNet _process method:

  def _process(
      self,
      latent_graph_0: typed_graph.TypedGraph,
      processor_networks: List[GraphToGraphNetwork],
  ) -> typed_graph.TypedGraph:
    # Processes the latent graph with several steps of message passing.

    # Do `num_message_passing_steps` with each of the `self._processor_networks`
    # with unshared weights, and repeat that `self._num_processor_repetitions`
    # times.
    latent_graph = latent_graph_0
    latent_graphs = [latent_graph_0] # store initial

    for unused_repetition_i in range(self._num_processor_repetitions):
      for processor_network in processor_networks:
        latent_graph = self._process_step(processor_network, latent_graph)
        latent_graphs.append(latent_graph)  # store after each step

    return latent_graphs
