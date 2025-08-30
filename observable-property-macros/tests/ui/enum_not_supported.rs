use observable_property::Observable;

// This should fail to compile because Observable can only be derived for structs
#[derive(Observable)]
enum BadEnum {
    Variant1,
    Variant2,
}

fn main() {}
