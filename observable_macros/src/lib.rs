use proc_macro::TokenStream;
use quote::{quote, format_ident};
use syn::{parse_macro_input, ItemStruct, DeriveInput, Data, Fields, Type};

// Attribute macro: #[observable]
#[proc_macro_attribute]
pub fn observable(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemStruct);

    // Only support named structs
    let struct_name = input.ident.clone();
    let vis = input.vis.clone();

    let fields = match input.fields {
        Fields::Named(fields_named) => fields_named.named,
        _ => {
            return syn::Error::new_spanned(&input, "observable can only be applied to structs with named fields")
                .to_compile_error()
                .into();
        }
    };

    let mut new_fields = Vec::new();
    let mut ctor_inits = Vec::new();
    let mut ctor_args = Vec::new();

    for field in fields.iter() {
        let name = field.ident.as_ref().unwrap();
        let ty = &field.ty;

        // Field-level control: #[observable] forces wrapping, #[no_observable] skips wrapping.
        let _has_obs_attr = field.attrs.iter().any(|a| a.path().is_ident("observable") || a.path().is_ident("observable_attr"));
        let has_noobs_attr = field.attrs.iter().any(|a| a.path().is_ident("no_observable") || a.path().is_ident("noob") || a.path().is_ident("no-observable"));

        if has_noobs_attr {
            // Keep original type
            new_fields.push(quote! { #name: #ty });
            ctor_args.push(quote! { #name: #ty });
            ctor_inits.push(quote! { #name });
        } else {
            // Wrap field (either forced or default)
            let wrapped_ty: Type = syn::parse_quote!(observable_property::ObservableProperty<#ty>);
            new_fields.push(quote! { #name: #wrapped_ty });
            ctor_args.push(quote! { #name: #ty });
            ctor_inits.push(quote! { #name: observable_property::ObservableProperty::new(#name) });
        }
    }

    let gen = quote! {
        #vis struct #struct_name {
            #( #new_fields, )*
        }

        impl #struct_name {
            pub fn new( #( #ctor_args ),* ) -> Self {
                Self {
                    #( #ctor_inits, )*
                }
            }
        }
    };

    gen.into()
}

// Derive macro: #[derive(Observable)]
#[proc_macro_derive(Observable)]
pub fn derive_observable(input: TokenStream) -> TokenStream {
    let input_parsed = parse_macro_input!(input as DeriveInput);

    let name = input_parsed.ident.clone();
    let vis = input_parsed.vis.clone();

    let fields = match &input_parsed.data {
        Data::Struct(data_struct) => match &data_struct.fields {
            Fields::Named(named) => named.named.clone(),
            _ => {
                return syn::Error::new_spanned(&input_parsed, "Observable derive only supports structs with named fields")
                    .to_compile_error()
                    .into();
            }
        },
        _ => {
            return syn::Error::new_spanned(&input_parsed, "Observable derive only supports structs")
                .to_compile_error()
                .into();
        }
    };

    // If any field has an explicit #[observable], then only wrap those fields.
    let explicit_obs_count = fields.iter().filter(|f| f.attrs.iter().any(|a| a.path().is_ident("observable") || a.path().is_ident("observable_attr"))).count();

    let observable_name = format_ident!("{}Observable", name);

    let mut obs_fields = Vec::new();
    let mut from_inits = Vec::new();

    for field in fields.iter() {
        let fname = field.ident.as_ref().unwrap();
        let fty = &field.ty;
        let has_obs_attr = field.attrs.iter().any(|a| a.path().is_ident("observable") || a.path().is_ident("observable_attr"));
        let has_noobs_attr = field.attrs.iter().any(|a| a.path().is_ident("no_observable") || a.path().is_ident("noob") || a.path().is_ident("no-observable"));

        // Decide whether to wrap this field.
        let should_wrap = if explicit_obs_count > 0 {
            // only wrap if field explicitly annotated
            has_obs_attr
        } else {
            // default: wrap unless marked with #[no_observable]
            !has_noobs_attr
        };

        if should_wrap {
            let wrapped: Type = syn::parse_quote!(observable_property::ObservableProperty<#fty>);
            obs_fields.push(quote! { pub #fname: #wrapped });
            from_inits.push(quote! { #fname: observable_property::ObservableProperty::new(orig.#fname) });
        } else {
            obs_fields.push(quote! { pub #fname: #fty });
            from_inits.push(quote! { #fname: orig.#fname });
        }
    }

    let gen = quote! {
        #vis struct #observable_name {
            #( #obs_fields, )*
        }

        impl #name {
            pub fn into_observable(self) -> #observable_name {
                let orig = self;
                #observable_name {
                    #( #from_inits, )*
                }
            }
        }

        impl From<#name> for #observable_name {
            fn from(orig: #name) -> Self {
                orig.into_observable()
            }
        }
    };

    gen.into()
}
