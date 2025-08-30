use observable_property::Observable;

// This should fail to compile because tuple structs aren't supported by #[derive(Observable)]
#[derive(Observable)]
struct BadStruct(String, i32);

fn main() {}
