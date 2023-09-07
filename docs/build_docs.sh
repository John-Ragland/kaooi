sphinx-apidoc -o "api/" --module-first --no-toc --force "../src/kaooi"
sphinx-build -b html . "_build/"