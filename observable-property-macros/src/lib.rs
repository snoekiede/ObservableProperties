extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields, Ident};

/// Creates an observable property from a value
///
/// This macro is a convenient way to wrap any value in an `ObservableProperty`
/// without explicitly calling `ObservableProperty::new()`.
///
/// # Examples
///
/// ```rust
/// use observable_property::observable;
/// use std::sync::Arc;
///
/// let counter = observable!(0);
/// let name = observable!("Alice".to_string());
///
/// counter.subscribe(Arc::new(|old, new| {
///     println!("Counter: {} -> {}", old, new);
/// })).unwrap();
///
/// counter.set(42).unwrap();
/// ```
#[proc_macro]
pub fn observable(input: TokenStream) -> TokenStream {
    let expr = parse_macro_input!(input as syn::Expr);

    // Try to use the external crate reference first, but fall back to internal
    let expanded = quote! {
        {
            #[allow(unused_imports)]
            use observable_property::ObservableProperty as __ObservableProperty;
            __ObservableProperty::new(#expr)
        }
    };

    TokenStream::from(expanded)
}

/// Derive macro for creating structs with observable fields
///
/// This macro generates getter and setter methods for fields marked with `#[observable]`.
/// The fields must be of type `ObservableProperty<T>`.
///
/// # Examples
///
/// ```rust
/// use observable_property::{Observable, observable};
///
/// #[derive(Observable)]
/// struct Person {
///     #[observable]
///     name: String,
///     #[observable]
///     age: i32,
///     id: u64, // Regular field
/// }
/// ```
#[proc_macro_derive(Observable, attributes(observable))]
pub fn derive_observable(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let generics = &input.generics;
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let fields = match &input.data {
        Data::Struct(data_struct) => match &data_struct.fields {
            Fields::Named(fields) => &fields.named,
            _ => {
                return syn::Error::new_spanned(
                    &input,
                    "Observable can only be derived for structs with named fields"
                ).to_compile_error().into();
            }
        },
        _ => {
            return syn::Error::new_spanned(
                &input,
                "Observable can only be derived for structs"
            ).to_compile_error().into();
        }
    };

    let mut methods = Vec::new();
    let mut constructor_params = Vec::new();
    let mut constructor_init = Vec::new();

    for field in fields {
        let field_name = field.ident.as_ref().unwrap();
        let field_type = &field.ty;

        // Check if field has #[observable] attribute
        let has_observable_attr = field.attrs.iter().any(|attr| {
            attr.path().is_ident("observable")
        });

        if has_observable_attr {
            // Extract the inner type from ObservableProperty<T>
            let inner_type = if let syn::Type::Path(type_path) = field_type {
                if let Some(segment) = type_path.path.segments.last() {
                    if segment.ident == "ObservableProperty" {
                        if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
                            if let Some(syn::GenericArgument::Type(inner)) = args.args.first() {
                                inner
                            } else {
                                return syn::Error::new_spanned(
                                    field_type,
                                    "ObservableProperty must have a type argument"
                                ).to_compile_error().into();
                            }
                        } else {
                            return syn::Error::new_spanned(
                                field_type,
                                "ObservableProperty must have a type argument"
                            ).to_compile_error().into();
                        }
                    } else {
                        return syn::Error::new_spanned(
                            field_type,
                            "Observable fields must be of type ObservableProperty<T>"
                        ).to_compile_error().into();
                    }
                } else {
                    return syn::Error::new_spanned(
                        field_type,
                        "Invalid field type"
                    ).to_compile_error().into();
                }
            } else {
                return syn::Error::new_spanned(
                    field_type,
                    "Observable fields must be of type ObservableProperty<T>"
                ).to_compile_error().into();
            };

            // Add parameter for constructor (raw type, not ObservableProperty)
            constructor_params.push(quote! { #field_name: #inner_type });
            // Add initialization (wrap in ObservableProperty)
            constructor_init.push(quote! {
                #field_name: observable_property::ObservableProperty::new(#field_name)
            });

            // Generate getter method
            let getter_name = Ident::new(&format!("get_{}", field_name), field_name.span());
            let getter = quote! {
                pub fn #getter_name(&self) -> Result<#inner_type, observable_property::PropertyError> {
                    self.#field_name.get()
                }
            };

            // Generate setter method
            let setter_name = Ident::new(&format!("set_{}", field_name), field_name.span());
            let setter = quote! {
                pub fn #setter_name(&self, value: #inner_type) -> Result<(), observable_property::PropertyError> {
                    self.#field_name.set(value)
                }
            };

            // Generate subscribe method
            let subscribe_name = Ident::new(&format!("subscribe_{}", field_name), field_name.span());
            let subscribe = quote! {
                pub fn #subscribe_name(&self, observer: std::sync::Arc<dyn Fn(&#inner_type, &#inner_type) + Send + Sync>) -> Result<observable_property::ObserverId, observable_property::PropertyError> {
                    self.#field_name.subscribe(observer)
                }
            };

            // Generate unsubscribe method
            let unsubscribe_name = Ident::new(&format!("unsubscribe_{}", field_name), field_name.span());
            let unsubscribe = quote! {
                pub fn #unsubscribe_name(&self, id: observable_property::ObserverId) -> Result<bool, observable_property::PropertyError> {
                    self.#field_name.unsubscribe(id)
                }
            };

            // Generate filtered subscribe method
            let subscribe_filtered_name = Ident::new(&format!("subscribe_{}_filtered", field_name), field_name.span());
            let subscribe_filtered = quote! {
                pub fn #subscribe_filtered_name<F>(&self, observer: std::sync::Arc<dyn Fn(&#inner_type, &#inner_type) + Send + Sync>, filter: F) -> Result<observable_property::ObserverId, observable_property::PropertyError>
                where
                    F: Fn(&#inner_type, &#inner_type) -> bool + Send + Sync + 'static,
                {
                    self.#field_name.subscribe_filtered(observer, filter)
                }
            };

            methods.push(getter);
            methods.push(setter);
            methods.push(subscribe);
            methods.push(unsubscribe);
            methods.push(subscribe_filtered);
        } else {
            // For non-observable fields, just add them to constructor
            constructor_params.push(quote! { #field_name: #field_type });
            constructor_init.push(quote! { #field_name });
        }
    }

    // Generate only the implementation, not the struct definition
    let expanded = quote! {
        impl #impl_generics #name #ty_generics #where_clause {
            /// Create a new instance with observable fields automatically wrapped
            pub fn new(#(#constructor_params),*) -> Self {
                Self {
                    #(#constructor_init),*
                }
            }

            #(#methods)*
        }
    };

    TokenStream::from(expanded)
}

/// Attribute macro for marking individual fields as observable
///
/// This is primarily used with the `Observable` derive macro to mark
/// which fields should have observable getter/setter methods generated.
///
/// # Examples
///
/// ```rust
/// use observable_property::{Observable, observable_field};
///
/// #[derive(Observable)]
/// struct Config {
///     #[observable]
///     debug_mode: bool,
///     #[observable]
///     max_connections: usize,
///     version: String, // Not observable
/// }
/// ```
#[proc_macro_attribute]
pub fn observable_field(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // This is mainly a marker attribute for the derive macro
    // We just pass through the item unchanged
    item
}
