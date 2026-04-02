import Capacitor
import UIKit
import WebKit

final class AppViewController: CAPBridgeViewController {
    override func viewDidLoad() {
        super.viewDidLoad()
        webView?.configuration.allowsInlineMediaPlayback = true
        if #available(iOS 10.0, *) {
            webView?.configuration.mediaTypesRequiringUserActionForPlayback = []
        }
    }

    override func capacitorDidLoad() {
        super.capacitorDidLoad()
        bridge?.registerPluginInstance(DiveSenseiMediaPlugin())
    }
}
