use super::memory_management::GraphMemoryManagement;
use crate::{
    NodeID,
    checkpoint::{
        base::{Checkpointer, NodeTree},
        builder::CheckpointerBuilder,
    },
    collections::HashMap,
    grads::Gradients,
    graph::{StepBoxed, traversal::BreadthFirstSearch},
    tensor::NodeRefCount,
};
use alloc::vec::Vec;

#[derive(Default)]
pub struct AutodiffServer {
    steps: HashMap<NodeID, StepBoxed>,
    actions_builder: HashMap<NodeID, CheckpointerBuilder>,
    memory_management: GraphMemoryManagement,
}

impl AutodiffServer {
    pub fn register(&mut self, rc: NodeRefCount, step: StepBoxed, actions: CheckpointerBuilder) {
        let parents = step.parents();
        let node_id = *rc.as_ref();

        self.memory_management.register(rc, parents);

        self.steps.insert(node_id, step);
        self.actions_builder.insert(node_id, actions);
    }

    pub fn backward(&mut self, grads: Gradients, node_id: NodeID) -> Gradients {
        let step = self.steps.remove(&node_id).expect(
            "Node should have a step registered, did you forget to call \
             `Tensor::register_grad` on the tensor where you need gradients?",
        );
        let builder = self.actions_builder.remove(&node_id).unwrap();

        let (tape, checkpointer) = self.build_tape(node_id, step, builder);

        let gradients = Self::execute_steps(tape, grads, checkpointer);

        // Cleanup
        self.memory_management
            .free_unavailable_nodes(|node_id: &NodeID| {
                self.steps.remove(node_id);
                self.actions_builder.remove(node_id);
            });

        gradients
    }

    fn build_tape(
        &mut self,
        node: NodeID,
        node_step: StepBoxed,
        mut builder: CheckpointerBuilder,
    ) -> (Vec<Vec<StepBoxed>>, Checkpointer) {
        let mut tape = (0..node_step.depth())
            .map(|_| Vec::with_capacity(1))
            .collect::<Vec<_>>();

        let mut tree = HashMap::default();

        BreadthFirstSearch.traverse(node, node_step, &mut self.steps, |id, step| {
            self.memory_management.consume_node(id);

            let depth = step.depth();

            if depth == 0 {
                return;
            }

            if let Some(steps) = tape.get_mut(depth - 1) {
                let parents = step.parents().into_iter().filter(|s| *s != id);
                tree.insert(id, parents.collect());
                steps.push(step);
            }

            if let Some(node_builder) = self.actions_builder.remove(&id) {
                builder.extend(node_builder);
            }
        });

        let checkpointer = builder.build(NodeTree::new(tree));

        (tape, checkpointer)
    }

    fn execute_steps(
        tape: Vec<Vec<StepBoxed>>,
        mut grads: Gradients,
        mut checkpointer: Checkpointer,
    ) -> Gradients {
        tape.into_iter().rev().for_each(|steps| {
            steps
                .into_iter()
                .for_each(|step| step.step(&mut grads, &mut checkpointer))
        });

        #[cfg(feature = "export_tests")]
        // For checkpointing tests
        assert!(checkpointer.is_empty());

        grads
    }
}
