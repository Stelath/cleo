fn main() {
    let proto_path = std::path::Path::new("../../../protos/frontend.proto");
    tonic_build::configure()
        .build_server(false)
        .compile_protos(&[proto_path], &["../../../protos"])
        .expect("failed to compile frontend.proto");
    tauri_build::build();
}
