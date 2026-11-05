import Metal
import Foundation
// Compile-only pipeline info of every kernel in a Metal source file: builds the library and a compute pipeline state
// per function (the GPU backend compile), no command buffers / nothing runs on the GPU. maxTotalThreadsPerThreadgroup
// drops when a kernel needs many registers per thread, so it is a register-pressure proxy.
let path = CommandLine.arguments[1]
let src = try! String(contentsOfFile: path, encoding: .utf8)
guard let dev = MTLCreateSystemDefaultDevice() else { print("NODEVICE"); exit(2) }
let opts = MTLCompileOptions()
opts.languageVersion = .version4_0
do {
    let lib = try dev.makeLibrary(source: src, options: opts)
    for name in lib.functionNames.sorted() {
        guard let fn = lib.makeFunction(name: name) else { continue }
        do {
            let pso = try dev.makeComputePipelineState(function: fn)
            print("PSO \(name) max_threads=\(pso.maxTotalThreadsPerThreadgroup) simd=\(pso.threadExecutionWidth) tgmem=\(pso.staticThreadgroupMemoryLength)")
        } catch {
            print("PSOERR \(name) \(error)")
        }
    }
} catch {
    print("ERR \(error)")
    exit(1)
}
