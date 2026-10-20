use std::collections::HashMap;

#[derive(Debug, Default)]
pub struct TypedArray {
    pub data: Vec<f32>,
    pub item_size: usize,
}

impl TypedArray {
    pub fn new(item_size: usize) -> Self {
        Self {
            data: Vec::new(),
            item_size,
        }
    }

    pub fn push_values(&mut self, values: &[f32]) {
        self.data.extend_from_slice(values);
    }
}

pub type AttrSet = HashMap<String, TypedArray>;

#[derive(Debug, Default)]
pub struct TypedAttributeData {
    pub group_attributes: Vec<AttrSet>,
    pub group_count: usize,
}

impl TypedAttributeData {
    pub fn new() -> Self {
        let mut s = Self::default();
        s.group_attributes.push(HashMap::new());
        s
    }

    pub fn initialize_array(&mut self, name: &str, item_size: usize) {
        for set in &mut self.group_attributes {
            set.entry(name.to_string())
                .or_insert_with(|| TypedArray::new(item_size));
        }
    }

    pub fn get_group_attr_set(&mut self, index: usize) -> &mut AttrSet {
        while index >= self.group_attributes.len() {
            let ref_set = &self.group_attributes[0];
            let mut new_set = HashMap::new();
            for (k, v) in ref_set {
                new_set.insert(k.clone(), TypedArray::new(v.item_size));
            }
            self.group_attributes.push(new_set);
        }
        self.group_count = self.group_count.max(index + 1);
        &mut self.group_attributes[index]
    }
}
