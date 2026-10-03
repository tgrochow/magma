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

          # shaderc-sys 0.8.3 uses uint32_t without including <cstdint> but we
          # need that on Ubuntu and we need Ubuntu because that's the only distro
          # available on GitHub Actions. This is not needed on NixOS.
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
