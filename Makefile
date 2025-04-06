MOD_ARCH := $(shell uname -m)
MOD_OS := $(shell uname -s)

GO_BUILD_ENV :=
GO_BUILD_FLAGS :=
MODULE_BINARY := bin/yolo-onnx

ifeq ($(VIAM_TARGET_OS), windows)
	GO_BUILD_ENV += GOOS=windows GOARCH=amd64
	GO_BUILD_FLAGS := -tags no_cgo	
	MODULE_BINARY = bin/yolo-onnx.exe
endif

$(MODULE_BINARY): Makefile go.mod *.go cmd/module/*.go 
	$(GO_BUILD_ENV) go build $(GO_BUILD_FLAGS) -o $(MODULE_BINARY) cmd/module/main.go

lint:
	gofmt -s -w .

update:
	go get go.viam.com/rdk@latest
	go mod tidy

test:
	go test ./...

# module.tar.gz: meta.json $(MODULE_BINARY)
# ifeq ($(VIAM_TARGET_OS), windows)
# 	jq '.entrypoint = "./bin/yolo-onnx.exe"' meta.json > temp.json && mv temp.json meta.json
# else
# 	strip $(MODULE_BINARY)
# endif
# 	tar czf $@ meta.json $(MODULE_BINARY) third_party/
# ifeq ($(VIAM_TARGET_OS), windows)
# 	git checkout meta.json
# endif

module.tar.gz: meta.json $(MODULE_BINARY)
ifeq ($(VIAM_TARGET_OS),windows) # this needs to be at the top since windows is emulated
	jq '.entrypoint = "./bin/yolo-onnx.exe"' meta.json > temp.json && mv temp.json meta.json
	tar -czf $@ $(MODULE_BINARY) third_party/onnxruntime.dll meta.json
else ifeq ($(MOD_OS),Darwin)
ifeq ($(MOD_ARCH),x86_64)
	@echo "Unsupported OS: $(MOD_OS) or architecture: $(MOD_ARCH)"
else ifeq ($(MOD_ARCH),arm64)
	strip $(MODULE_BINARY)
	tar -czf $@ $(MODULE_BINARY) third_party/onnxruntime_arm64.dylib
endif
else ifeq ($(MOD_OS),Linux)
ifeq ($(MOD_ARCH),x86_64)
	strip $(MODULE_BINARY)
	tar -czf $@ $(MODULE_BINARY) third_party/onnxruntime.so
else ifeq ($(MOD_ARCH),arm64)
	strip $(MODULE_BINARY)
	tar -czf $@ $(MODULE_BINARY) third_party/onnxruntime_arm64.so
else ifeq ($(MOD_ARCH),aarch64)
	strip $(MODULE_BINARY)
	tar -czf $@ $(MODULE_BINARY) third_party/onnxruntime_arm64.so
endif
else
	@echo "Unsupported OS: $(MOD_OS) or architecture: $(MOD_ARCH)"
endif
ifeq ($(VIAM_TARGET_OS), windows)
	git checkout meta.json
endif

module: test module.tar.gz

all: test module.tar.gz

setup:
	go mod tidy
