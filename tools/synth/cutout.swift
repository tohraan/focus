import Vision
import AppKit
// usage: cutout in.png out.png  -> RGBA with Vision person matte as alpha
let a = CommandLine.arguments
let url = URL(fileURLWithPath: a[1])
let req = VNGeneratePersonSegmentationRequest(); req.qualityLevel = .accurate; req.outputPixelFormat = kCVPixelFormatType_OneComponent8
let h = VNImageRequestHandler(url: url); try! h.perform([req])
let mask = (req.results!.first as! VNPixelBufferObservation).pixelBuffer
let src = CIImage(contentsOf: url)!
var m = CIImage(cvPixelBuffer: mask)
m = m.transformed(by: CGAffineTransform(scaleX: src.extent.width / m.extent.width, y: src.extent.height / m.extent.height))
let out = src.applyingFilter("CIBlendWithMask", parameters: [kCIInputBackgroundImageKey: CIImage(color: .clear).cropped(to: src.extent), kCIInputMaskImageKey: m])
let ctx = CIContext()
try! ctx.writePNGRepresentation(of: out, to: URL(fileURLWithPath: a[2]), format: .RGBA8, colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)
