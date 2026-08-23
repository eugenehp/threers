use std::collections::HashMap;

#[derive(Debug, Default)]
pub struct IntersectionMap {
    pub intersection_set: HashMap<usize, Vec<usize>>,
    pub ids: Vec<usize>,
}

impl IntersectionMap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, id: usize, intersection_id: usize) {
        if let std::collections::hash_map::Entry::Vacant(e) = self.intersection_set.entry(id) {
            e.insert(Vec::new());
            self.ids.push(id);
        }
        self.intersection_set
            .get_mut(&id)
            .unwrap()
            .push(intersection_id);
    }
}
