//! PowerShell Constrained Language Mode allowed types (adapted from clmTypes.ts).
//!
//! CLM restricts .NET type usage to this allowlist under AppLocker/WDAC
//! lockdown. We invert it: any type literal NOT in this set → ask. Microsoft
//! maintains the list; one canonical check replaces enumerating dangerous types.
//!
//! SECURITY: adsi/adsisearcher/wmi/wmiclass/wmisearcher/cimsession (and their FQ
//! names) are deliberately REMOVED from Microsoft's list — their casts perform
//! network binds (LDAP/remote WMI) against unvalidated targets. Do not re-add.
//! Entries are lowercase; both short and full names are stored (AST emits either).

use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::HashSet;

pub static CLM_ALLOWED_TYPES: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    [
        "alias", "allowemptycollection", "allowemptystring", "allownull",
        "argumentcompleter", "argumentcompletions", "array", "bigint", "bool", "byte",
        "char", "cimclass", "cimconverter", "ciminstance", "cimtype", "cmdletbinding",
        "cultureinfo", "datetime", "decimal", "double", "dsclocalconfigurationmanager",
        "dscproperty", "dscresource", "experimentaction", "experimental",
        "experimentalfeature", "float", "guid", "hashtable", "int", "int16", "int32",
        "int64", "ipaddress", "ipendpoint", "long", "mailaddress", "norunspaceaffinity",
        "nullstring", "objectsecurity", "ordered", "outputtype", "parameter",
        "physicaladdress", "pscredential", "pscustomobject", "psdefaultvalue",
        "pslistmodifier", "psobject", "psprimitivedictionary", "pstypenameattribute",
        "ref", "regex", "sbyte", "securestring", "semver", "short", "single", "string",
        "supportswildcards", "switch", "timespan", "uint", "uint16", "uint32", "uint64",
        "ulong", "uri", "ushort", "validatecount", "validatedrive", "validatelength",
        "validatenotnull", "validatenotnullorempty", "validatenotnullorwhitespace",
        "validatepattern", "validaterange", "validatescript", "validateset",
        "validatetrusteddata", "validateuserdrive", "version", "void", "wildcardpattern",
        "x500distinguishedname", "x509certificate", "xml", "system.array",
        "system.boolean", "system.byte", "system.char", "system.datetime",
        "system.decimal", "system.double", "system.guid", "system.int16", "system.int32",
        "system.int64", "system.numerics.biginteger", "system.sbyte", "system.single",
        "system.string", "system.timespan", "system.uint16", "system.uint32",
        "system.uint64", "system.uri", "system.version", "system.void",
        "system.collections.hashtable", "system.text.regularexpressions.regex",
        "system.globalization.cultureinfo", "system.net.ipaddress", "system.net.ipendpoint",
        "system.net.mail.mailaddress",
        "system.net.networkinformation.physicaladdress", "system.security.securestring",
        "system.security.cryptography.x509certificates.x509certificate",
        "system.security.cryptography.x509certificates.x500distinguishedname",
        "system.xml.xmldocument", "system.management.automation.pscredential",
        "system.management.automation.pscustomobject",
        "system.management.automation.pslistmodifier", "system.management.automation.psobject",
        "system.management.automation.psprimitivedictionary",
        "system.management.automation.psreference",
        "system.management.automation.semanticversion",
        "system.management.automation.switchparameter",
        "system.management.automation.wildcardpattern",
        "system.management.automation.language.nullstring",
        "microsoft.management.infrastructure.cimclass",
        "microsoft.management.infrastructure.cimconverter",
        "microsoft.management.infrastructure.ciminstance",
        "microsoft.management.infrastructure.cimtype",
        "system.collections.specialized.ordereddictionary",
        "system.security.accesscontrol.objectsecurity", "object", "system.object",
        "microsoft.powershell.commands.modulespecification",
    ]
    .into_iter()
    .collect()
});

static ARRAY_SUFFIX_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\[\]$").unwrap());
static GENERIC_SUFFIX_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\[.*\]$").unwrap());

/// Normalize an AST TypeName: lowercase, strip array suffix and generic args.
pub fn normalize_type_name(name: &str) -> String {
    let n = name.to_lowercase();
    let n = ARRAY_SUFFIX_RE.replace(&n, ""); // "string[]" -> "string"
    let n = GENERIC_SUFFIX_RE.replace(&n, ""); // "list[int]" -> "list"
    n.trim().to_string()
}

/// True if type_name is in Microsoft's CLM allowlist (types outside → ask).
pub fn is_clm_allowed_type(type_name: &str) -> bool {
    CLM_ALLOWED_TYPES.contains(normalize_type_name(type_name).as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allowed_and_disallowed() {
        assert!(is_clm_allowed_type("string"));
        assert!(is_clm_allowed_type("String[]"));
        assert!(is_clm_allowed_type("System.Net.IPAddress"));
        // deliberately removed network-bind types
        assert!(!is_clm_allowed_type("adsi"));
        assert!(!is_clm_allowed_type("wmiclass"));
        assert!(!is_clm_allowed_type("System.Diagnostics.Process"));
    }
}
