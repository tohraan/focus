import Foundation
import CoreImage
import CoreGraphics

/// The "other world": calm natural scenery (mountains, waterfall, forest), painted once as depth-layered
/// landscape.  If the bundle contains world_<name>.jpg (e.g. a licensed photo) that image is used instead.
final class IsolationWorld {
    enum Theme: Int, CaseIterable {
        case moon, mountains, waterfall, forest
        var name: String { ["Moon", "Mountains", "Waterfall", "Forest"][rawValue] }
        var file: String { ["world_moon", "world_mountains", "world_waterfall", "world_forest"][rawValue] }
    }

    let size: CGSize
    private(set) var theme: Theme
    private var image: CIImage

    init(size: CGSize, theme: Theme = .moon) {
        self.size = size; self.theme = theme
        image = IsolationWorld.make(size: size, theme: theme)
    }
    func setTheme(_ t: Theme) { theme = t; image = IsolationWorld.make(size: size, theme: t) }

    /// Static scene (no animation, no parallax): quiet by design.
    func frame(time: Double = 0, parallax: CGPoint = .zero) -> CIImage { image }

    // MARK: building
    private static func make(size: CGSize, theme: Theme) -> CIImage {
        for ext in ["jpg", "png"] {
            if let u = Bundle.main.url(forResource: theme.file, withExtension: ext), let ci = CIImage(contentsOf: u) {
                let s = max(size.width / ci.extent.width, size.height / ci.extent.height)
                let sc = ci.transformed(by: CGAffineTransform(scaleX: s, y: s))
                let dx = (sc.extent.width - size.width) / 2, dy = (sc.extent.height - size.height) / 2
                return sc.transformed(by: CGAffineTransform(translationX: -sc.extent.minX - dx, y: -sc.extent.minY - dy)).cropped(to: CGRect(origin: .zero, size: size))
            }
        }
        return theme == .moon ? paintedMoon(size: size) : painted(size: size, theme: theme)
    }

    private struct RNG { var s: UInt64
        mutating func next() -> CGFloat { s = s &* 6364136223846793005 &+ 1442695040888963407; return CGFloat((s >> 33) & 0xFFFFFF) / 16777215 } }

    private static func layer(_ size: CGSize, _ draw: (CGContext, CGColorSpace) -> Void) -> CIImage {
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        let c = CGContext(data: nil, width: Int(size.width), height: Int(size.height), bitsPerComponent: 8, bytesPerRow: 0, space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        draw(c, cs)
        return CIImage(cgImage: c.makeImage()!)
    }
    private static func col(_ cs: CGColorSpace, _ r: CGFloat, _ g: CGFloat, _ b: CGFloat, _ a: CGFloat = 1) -> CGColor { CGColor(colorSpace: cs, components: [r, g, b, a])! }
    private static func blur(_ i: CIImage, _ s: Double) -> CIImage { i.clampedToExtent().applyingGaussianBlur(sigma: s).cropped(to: i.extent) }

    /// Midpoint-displacement ridge: y values (0...1) at `n` columns.
    private static func ridge(_ n: Int, rough: CGFloat, rng: inout RNG, base: CGFloat, amp: CGFloat) -> [CGFloat] {
        var h = [CGFloat](repeating: 0, count: n + 1)
        h[0] = rng.next(); h[n] = rng.next()
        var step = n, scale: CGFloat = 1
        while step > 1 {
            var i = step / 2
            while i < n { h[i] = (h[i - step / 2] + h[min(n, i + step / 2)]) / 2 + (rng.next() - 0.5) * scale; i += step }
            step /= 2; scale *= rough
        }
        let lo = h.min()!, hi = h.max()!
        return h.map { base + amp * ($0 - lo) / max(hi - lo, 1e-6) }
    }

    /// The moon in space: starfield, a large lit moon with maria and craters, and a dark lunar ridge in the foreground.
    private static func paintedMoon(size: CGSize) -> CIImage {
        let W = size.width, H = size.height, rect = CGRect(origin: .zero, size: size)
        var rng = RNG(s: 0x4D4F4F4E)
        var img = layer(size) { c, cs in
            let g = CGGradient(colorsSpace: cs, colors: [col(cs, 0.03, 0.04, 0.10), col(cs, 0.005, 0.005, 0.02)] as CFArray, locations: [0, 1])!
            c.drawLinearGradient(g, start: CGPoint(x: 0, y: 0), end: CGPoint(x: 0, y: H), options: [.drawsBeforeStartLocation, .drawsAfterEndLocation])
        }
        img = layer(size) { c, cs in      // stars
            for _ in 0..<320 {
                let x = rng.next() * W, y = rng.next() * H, r = 0.6 + 1.6 * rng.next() * rng.next(), a = 0.35 + 0.65 * rng.next()
                c.setFillColor(col(cs, 1, 1, 0.95, a)); c.fillEllipse(in: CGRect(x: x - r, y: y - r, width: 2 * r, height: 2 * r))
            }
        }.composited(over: img)
        let mc = CGPoint(x: W * 0.60, y: H * 0.66), mr = W * 0.34
        img = blur(layer(size) { c, cs in     // glow
            c.setFillColor(col(cs, 0.75, 0.8, 1, 0.22)); c.fillEllipse(in: CGRect(x: mc.x - mr * 1.25, y: mc.y - mr * 1.25, width: mr * 2.5, height: mr * 2.5))
        }, 40).composited(over: img)
        img = layer(size) { c, cs in          // moon disc, lit from the upper left
            c.saveGState(); c.addEllipse(in: CGRect(x: mc.x - mr, y: mc.y - mr, width: 2 * mr, height: 2 * mr)); c.clip()
            let g = CGGradient(colorsSpace: cs, colors: [col(cs, 0.97, 0.96, 0.92), col(cs, 0.74, 0.73, 0.70), col(cs, 0.30, 0.30, 0.31)] as CFArray, locations: [0, 0.6, 1])!
            c.drawRadialGradient(g, startCenter: CGPoint(x: mc.x - mr * 0.35, y: mc.y + mr * 0.35), startRadius: 0, endCenter: CGPoint(x: mc.x - mr * 0.1, y: mc.y + mr * 0.1), endRadius: mr * 1.25, options: [.drawsAfterEndLocation])
            for _ in 0..<7 {   // maria: broad dark patches
                let a = rng.next() * 2 * .pi, d = rng.next() * mr * 0.6, w = mr * (0.25 + 0.3 * rng.next())
                c.setFillColor(col(cs, 0.42, 0.42, 0.45, 0.32)); c.fillEllipse(in: CGRect(x: mc.x + cos(a) * d - w / 2, y: mc.y + sin(a) * d - w / 2, width: w, height: w * 0.8))
            }
            for _ in 0..<45 {  // craters: dark floor, bright rim
                let a = rng.next() * 2 * .pi, d = sqrt(rng.next()) * mr * 0.92, r = mr * (0.012 + 0.05 * rng.next() * rng.next())
                let r0 = CGRect(x: mc.x + cos(a) * d - r, y: mc.y + sin(a) * d - r, width: 2 * r, height: 2 * r)
                c.setFillColor(col(cs, 0.35, 0.35, 0.37, 0.45)); c.fillEllipse(in: r0)
                c.setStrokeColor(col(cs, 1, 1, 1, 0.35)); c.setLineWidth(1.5); c.strokeEllipse(in: r0.offsetBy(dx: -r * 0.12, dy: r * 0.12))
            }
            c.restoreGState()
        }.composited(over: img)
        let ridgeA = ridge(256, rough: 0.55, rng: &rng, base: 0.10, amp: 0.07)
        let ridgeB = ridge(256, rough: 0.5, rng: &rng, base: 0.03, amp: 0.05)
        for (h, tone) in [(ridgeA, 0.20), (ridgeB, 0.10)] {
            img = blur(layer(size) { c, cs in
                let p = CGMutablePath(); p.move(to: .zero)
                for i in 0...256 { p.addLine(to: CGPoint(x: W * CGFloat(i) / 256, y: H * h[i])) }
                p.addLine(to: CGPoint(x: W, y: 0)); p.closeSubpath()
                c.addPath(p); c.setFillColor(col(cs, CGFloat(tone), CGFloat(tone), CGFloat(tone) * 1.05)); c.fillPath()
            }, 1.0).composited(over: img)
        }
        let vig = CIFilter(name: "CIVignette", parameters: [kCIInputImageKey: img, kCIInputIntensityKey: 0.6, kCIInputRadiusKey: 1.8])!.outputImage!
        return vig.cropped(to: rect)
    }

    private static func painted(size: CGSize, theme: Theme) -> CIImage {
        let W = size.width, H = size.height, rect = CGRect(origin: .zero, size: size)
        var rng = RNG(s: UInt64(theme.rawValue + 7) &* 0x9E3779B97F4A7C15)
        // sky
        let skyTop: (CGFloat, CGFloat, CGFloat) = theme == .forest ? (0.62, 0.78, 0.86) : (0.40, 0.62, 0.86)
        let skyLow: (CGFloat, CGFloat, CGFloat) = theme == .forest ? (0.93, 0.95, 0.88) : (0.86, 0.92, 0.96)
        var img = layer(size) { c, cs in
            let g = CGGradient(colorsSpace: cs, colors: [col(cs, skyLow.0, skyLow.1, skyLow.2), col(cs, skyTop.0, skyTop.1, skyTop.2)] as CFArray, locations: [0, 1])!
            c.drawLinearGradient(g, start: CGPoint(x: 0, y: H * 0.55), end: CGPoint(x: 0, y: H), options: [.drawsBeforeStartLocation, .drawsAfterEndLocation])
        }
        // soft clouds
        let clouds = blur(layer(size) { c, cs in
            for _ in 0..<9 {
                let x = rng.next() * W, y = H * (0.68 + 0.28 * rng.next()), w = W * (0.25 + 0.3 * rng.next())
                c.setFillColor(col(cs, 1, 1, 1, 0.5)); c.fillEllipse(in: CGRect(x: x - w / 2, y: y, width: w, height: w * 0.16))
                c.fillEllipse(in: CGRect(x: x - w / 3, y: y + w * 0.05, width: w * 0.6, height: w * 0.14))
            }
        }, 12)
        img = clouds.composited(over: img)

        // distant mountains (snow-capped, hazy) -> mid ridges -> hills; each layer a bit darker/greener/sharper
        struct L { var base: CGFloat; var amp: CGFloat; var rough: CGFloat; var c: (CGFloat, CGFloat, CGFloat); var haze: CGFloat; var blur: Double; var snow: Bool }
        let layers: [L]
        switch theme {
        case .moon, .mountains:
            layers = [L(base: 0.50, amp: 0.30, rough: 0.62, c: (0.55, 0.62, 0.74), haze: 0.35, blur: 2.5, snow: true),
                      L(base: 0.42, amp: 0.20, rough: 0.58, c: (0.38, 0.48, 0.58), haze: 0.25, blur: 1.5, snow: false),
                      L(base: 0.30, amp: 0.14, rough: 0.55, c: (0.20, 0.34, 0.30), haze: 0.12, blur: 0.8, snow: false)]
        case .waterfall:
            layers = [L(base: 0.50, amp: 0.22, rough: 0.6, c: (0.50, 0.60, 0.68), haze: 0.35, blur: 2.5, snow: false),
                      L(base: 0.38, amp: 0.18, rough: 0.56, c: (0.26, 0.40, 0.34), haze: 0.2, blur: 1.2, snow: false)]
        case .forest:
            layers = [L(base: 0.45, amp: 0.16, rough: 0.55, c: (0.60, 0.70, 0.66), haze: 0.5, blur: 3, snow: false),
                      L(base: 0.36, amp: 0.14, rough: 0.55, c: (0.38, 0.52, 0.42), haze: 0.35, blur: 2, snow: false)]
        }
        let n = 256
        for l in layers {
            let h = ridge(n, rough: l.rough, rng: &rng, base: l.base, amp: l.amp)
            var lay = layer(size) { c, cs in
                let p = CGMutablePath(); p.move(to: CGPoint(x: 0, y: 0))
                for i in 0...n { p.addLine(to: CGPoint(x: W * CGFloat(i) / CGFloat(n), y: H * h[i])) }
                p.addLine(to: CGPoint(x: W, y: 0)); p.closeSubpath()
                c.saveGState(); c.addPath(p); c.clip()
                let top = l.snow ? col(cs, 0.96, 0.97, 1) : col(cs, l.c.0 * 1.15, l.c.1 * 1.15, l.c.2 * 1.1)
                let g = CGGradient(colorsSpace: cs, colors: [col(cs, l.c.0 + l.haze * 0.3, l.c.1 + l.haze * 0.3, l.c.2 + l.haze * 0.25), col(cs, l.c.0, l.c.1, l.c.2), top] as CFArray, locations: [0, 0.55, 1])!
                let hb = H * (l.base), ht = H * (l.base + l.amp)
                c.drawLinearGradient(g, start: CGPoint(x: 0, y: hb), end: CGPoint(x: 0, y: ht), options: [.drawsBeforeStartLocation, .drawsAfterEndLocation])
                if l.snow {   // snow only on the upper third of the relief
                    let sg = CGGradient(colorsSpace: cs, colors: [col(cs, 1, 1, 1, 0), col(cs, 1, 1, 1, 0.95)] as CFArray, locations: [0, 1])!
                    c.drawLinearGradient(sg, start: CGPoint(x: 0, y: hb + (ht - hb) * 0.66), end: CGPoint(x: 0, y: ht), options: [.drawsAfterEndLocation])
                }
                c.restoreGState()
            }
            lay = blur(lay, l.blur)
            img = lay.composited(over: img)
            // atmospheric haze between layers
            let haze = layer(size) { c, cs in
                let g = CGGradient(colorsSpace: cs, colors: [col(cs, 0.9, 0.95, 0.97, 0), col(cs, 0.9, 0.95, 0.97, l.haze)] as CFArray, locations: [0, 1])!
                c.drawLinearGradient(g, start: CGPoint(x: 0, y: H * (l.base + l.amp * 0.6)), end: CGPoint(x: 0, y: H * (l.base - 0.12)), options: [.drawsBeforeStartLocation, .drawsAfterEndLocation])
            }
            img = haze.composited(over: img)
        }

        // water (lake / pool) in the lower third with a soft reflection of what is above it
        let waterTop = H * (theme == .forest ? 0.0 : 0.27)
        if theme != .forest {
            let refl = blur(img.transformed(by: CGAffineTransform(translationX: 0, y: 2 * waterTop).scaledBy(x: 1, y: -1)).cropped(to: CGRect(x: 0, y: 0, width: W, height: waterTop)), 6)
                .applyingFilter("CIColorMatrix", parameters: ["inputRVector": CIVector(x: 0.72, y: 0, z: 0, w: 0), "inputGVector": CIVector(x: 0, y: 0.78, z: 0, w: 0), "inputBVector": CIVector(x: 0, y: 0, z: 0.78, w: 0), "inputAVector": CIVector(x: 0, y: 0, z: 0, w: 1)])
            let tint = layer(size) { c, cs in
                let g = CGGradient(colorsSpace: cs, colors: [col(cs, 0.18, 0.40, 0.42, 0.35), col(cs, 0.05, 0.18, 0.22, 0.7)] as CFArray, locations: [0, 1])!
                c.clip(to: CGRect(x: 0, y: 0, width: W, height: waterTop))
                c.drawLinearGradient(g, start: CGPoint(x: 0, y: waterTop), end: CGPoint(x: 0, y: 0), options: [])
            }
            img = tint.composited(over: refl.composited(over: img))
        }

        // waterfall: dark cliff on the right with a long white fall, mist at the base
        if theme == .waterfall {
            let cliff = layer(size) { c, cs in
                let rid = ridge(40, rough: 0.6, rng: &rng, base: 0.62, amp: 0.10)
                let p = CGMutablePath(); p.move(to: CGPoint(x: W * 0.40, y: waterTop)); p.addLine(to: CGPoint(x: W * 0.43, y: H * 0.70))
                for i in 0...40 { p.addLine(to: CGPoint(x: W * (0.43 + 0.57 * CGFloat(i) / 40), y: H * (rid[i] + (i < 8 ? 0.10 - CGFloat(i) * 0.012 : 0)))) }
                p.addLine(to: CGPoint(x: W, y: waterTop)); p.closeSubpath()
                c.addPath(p); c.clip()
                let g = CGGradient(colorsSpace: cs, colors: [col(cs, 0.10, 0.12, 0.12), col(cs, 0.30, 0.31, 0.29)] as CFArray, locations: [0, 1])!
                c.drawLinearGradient(g, start: CGPoint(x: 0, y: waterTop), end: CGPoint(x: W, y: H * 0.8), options: [])
                for _ in 0..<60 {   // rock strata
                    let x = W * (0.4 + 0.6 * rng.next()), y = waterTop + (H * 0.55) * rng.next()
                    c.setStrokeColor(col(cs, 0.05, 0.06, 0.06, 0.35)); c.setLineWidth(1 + 2 * rng.next())
                    c.move(to: CGPoint(x: x, y: y)); c.addLine(to: CGPoint(x: x + 40 * rng.next(), y: y - 10 * rng.next())); c.strokePath()
                }
            }
            img = blur(cliff, 0.8).composited(over: img)
            let top = H * 0.70, fx = W * 0.56, bottom = waterTop + H * 0.02
            let fall = layer(size) { c, cs in
                let p = CGMutablePath(); p.move(to: CGPoint(x: fx - W * 0.035, y: top)); p.addLine(to: CGPoint(x: fx + W * 0.035, y: top))
                p.addLine(to: CGPoint(x: fx + W * 0.075, y: bottom)); p.addLine(to: CGPoint(x: fx - W * 0.075, y: bottom)); p.closeSubpath()
                c.saveGState(); c.addPath(p); c.clip()
                let g = CGGradient(colorsSpace: cs, colors: [col(cs, 1, 1, 1, 0.95), col(cs, 0.82, 0.92, 0.98, 0.7)] as CFArray, locations: [0, 1])!
                c.drawLinearGradient(g, start: CGPoint(x: 0, y: top), end: CGPoint(x: 0, y: bottom), options: [])
                for _ in 0..<40 {   // streaks
                    let x = fx + (rng.next() - 0.5) * W * 0.14
                    c.setStrokeColor(col(cs, 1, 1, 1, 0.5 * rng.next())); c.setLineWidth(1 + 3 * rng.next())
                    c.move(to: CGPoint(x: x, y: top)); c.addLine(to: CGPoint(x: x + (x - fx) * 0.4, y: bottom)); c.strokePath()
                }
                c.restoreGState()
            }
            img = blur(fall, 1.2).composited(over: img)
            let mist = blur(layer(size) { c, cs in
                c.setFillColor(col(cs, 1, 1, 1, 0.55)); c.fillEllipse(in: CGRect(x: fx - W * 0.22, y: bottom - H * 0.04, width: W * 0.44, height: H * 0.10))
                c.setFillColor(col(cs, 1, 1, 1, 0.35)); c.fillEllipse(in: CGRect(x: fx - W * 0.3, y: bottom - H * 0.07, width: W * 0.6, height: H * 0.14))
            }, 18)
            img = mist.composited(over: img)
        }

        // conifer forest bands (far = lighter + blurred, near = dark + sharp)
        func forest(_ baseY: CGFloat, _ hMin: CGFloat, _ hMax: CGFloat, _ tone: CGFloat, _ blurS: Double, count: Int) -> CIImage {
            blur(layer(size) { c, cs in
                for i in 0..<count {
                    let x = W * (CGFloat(i) + rng.next() * 0.8) / CGFloat(count), h = H * (hMin + (hMax - hMin) * rng.next()), w = h * (0.28 + 0.1 * rng.next())
                    let y0 = H * baseY + (rng.next() - 0.5) * H * 0.02
                    let t = tone * (0.85 + 0.3 * rng.next())
                    for k in 0..<5 {   // tiers
                        let f = CGFloat(k) / 5
                        let ty = y0 + h * (0.15 + 0.85 * f), half = w * (1 - f) * 0.5 + w * 0.1, th = h * 0.24
                        let p = CGMutablePath(); p.move(to: CGPoint(x: x - half, y: ty)); p.addLine(to: CGPoint(x: x, y: ty + th)); p.addLine(to: CGPoint(x: x + half, y: ty)); p.closeSubpath()
                        c.setFillColor(col(cs, 0.10 * t, 0.30 * t + 0.02, 0.17 * t)); c.addPath(p); c.fillPath()
                    }
                    c.setFillColor(col(cs, 0.08, 0.06, 0.04)); c.fill(CGRect(x: x - w * 0.03, y: y0, width: w * 0.06, height: h * 0.2))
                }
                c.setFillColor(col(cs, 0.10 * tone, 0.28 * tone + 0.02, 0.15 * tone)); c.fill(CGRect(x: 0, y: 0, width: W, height: H * baseY + 4))
            }, blurS)
        }
        switch theme {
        case .moon, .mountains:
            img = forest(0.20, 0.07, 0.12, 0.8, 1.0, count: 38).composited(over: img)
            img = forest(0.05, 0.14, 0.22, 0.6, 0.0, count: 16).composited(over: img)
        case .waterfall:
            img = forest(0.20, 0.06, 0.10, 0.9, 1.5, count: 40).composited(over: img.applyingFilter("CIColorControls", parameters: [:]))
            img = forest(0.03, 0.12, 0.2, 0.55, 0.0, count: 14).composited(over: img)
        case .forest:
            img = forest(0.30, 0.10, 0.16, 1.2, 2.5, count: 46).composited(over: img)
            img = forest(0.18, 0.16, 0.26, 0.9, 1.2, count: 28).composited(over: img)
            // soft light shafts
            let rays = blur(layer(size) { c, cs in
                for i in 0..<6 {
                    let x = W * (0.1 + 0.16 * CGFloat(i))
                    let p = CGMutablePath(); p.move(to: CGPoint(x: x, y: H)); p.addLine(to: CGPoint(x: x + 30, y: H)); p.addLine(to: CGPoint(x: x - 140, y: 0)); p.addLine(to: CGPoint(x: x - 190, y: 0)); p.closeSubpath()
                    c.setFillColor(col(cs, 1, 0.97, 0.8, 0.10)); c.addPath(p); c.fillPath()
                }
            }, 12)
            img = rays.composited(over: img)
            img = forest(0.02, 0.22, 0.34, 0.5, 0.0, count: 12).composited(over: img)
        }

        // foreground greenery: grass blades and broad leaves along the bottom
        let grass = layer(size) { c, cs in
            for i in 0..<420 {
                let x = W * CGFloat(i) / 420 + (rng.next() - 0.5) * 8, h = H * (0.025 + 0.07 * rng.next() * rng.next() * 2), lean = (rng.next() - 0.5) * 36
                let t = 0.6 + 0.6 * rng.next()
                c.setStrokeColor(col(cs, 0.10 * t, 0.34 * t, 0.12 * t)); c.setLineWidth(2 + 2.5 * rng.next()); c.setLineCap(.round)
                c.move(to: CGPoint(x: x, y: 0)); c.addQuadCurve(to: CGPoint(x: x + lean, y: h), control: CGPoint(x: x + lean * 0.2, y: h * 0.55)); c.strokePath()
            }
            c.setFillColor(col(cs, 0.07, 0.22, 0.09)); c.fill(CGRect(x: 0, y: 0, width: W, height: H * 0.018))
        }
        img = grass.composited(over: img)
        // gentle vignette and warm grade so it reads as a photo, not an illustration
        let vig = CIFilter(name: "CIVignette", parameters: [kCIInputImageKey: img, kCIInputIntensityKey: 0.5, kCIInputRadiusKey: 1.6])!.outputImage!
        return vig.cropped(to: rect)
    }
}
