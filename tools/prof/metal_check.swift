import Metal
import Foundation
// Compile-only check of a Metal source file (no command buffers, nothing runs on the GPU).
let path = CommandLine.arguments[1]
let src = try! String(contentsOfFile: path, encoding: .utf8)
guard let dev = MTLCreateSystemDefaultDevice() else { print("NODEVICE"); exit(2) }
let opts = MTLCompileOptions()
opts.languageVersion = .version4_0
do {
    let lib = try dev.makeLibrary(source: src, options: opts)
    print("OK", lib.functionNames.count)
} catch {
    print("ERR \(error)")
    exit(1)
}
