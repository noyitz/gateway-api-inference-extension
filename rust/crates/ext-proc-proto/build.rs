fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto_root = "../../proto";

    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(
            &[
                &format!("{}/envoy/service/ext_proc/v3/external_processor.proto", proto_root),
            ],
            &[
                proto_root,
                &format!("{}/xds", proto_root),
            ],
        )?;

    Ok(())
}
