{
  description = "Magma development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };
      in
      {

        formatter = pkgs.nixfmt-tree;
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            cargo
            rustc
            cmake
            pkg-config

            vulkan-headers
            vulkan-loader
            vulkan-validation-layers

            wayland
            libxkbcommon
            xkeyboard-config
          ];

          CMAKE_POLICY_VERSION_MINIMUM = "3.5";
          CXXFLAGS = "-include cstdint";
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.vulkan-loader
            pkgs.wayland
            pkgs.libxkbcommon
          ];

          XKB_CONFIG_ROOT = "${pkgs.xkeyboard-config}/share/X11/xkb";

          SYSTEMD_XKB_DIRECTORY = "${pkgs.xkeyboard-config}/share/X11/xkb";
        };
      }
    );
}
