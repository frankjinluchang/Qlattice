Common command:

To build without cuda

$ time nix-build -j8 --cores 24

or

$ time nix-build -j8 --cores 24 default.nix

To build with cuda

$ time nix-build -j8 --cores 24 default-cuda.nix

To start a shell without cuda

$ time nix-shell

or

$ time nix-shell shell.nix

To start a shell with cuda

$ time nix-shell shell.nix --argstr name "qlat-jhub-sh-cuda"

To start a shell with only qlat

$ time nix-shell shell.nix --argstr name "qlat-sh"

To build (and run its unittest install check for) the qlat-ref reference package

$ time nix-build q-pkgs.nix -A pkgs-std-ucxless.qlat_ref

To run the qlat-ref tests directly from a checkout

$ cd ../qlat-ref && python3 -m unittest discover -s tests -t . -v

To start a shell with Filesystem Hierarchy Standard (FHS)

$ time nix-shell shell-fhs.nix

To start a shell with Filesystem Hierarchy Standard (FHS) and cuda

$ time nix-shell shell-fhs.nix --argstr name "qlat-jhub-fhs-cuda"
