// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { IDiamondCut } from "../interfaces/IDiamondCut.sol";

/*
 * =============== LibSmartSolve.sol ===============
 *
 * Defines a special storage layout (DiamondStorage) for all diamonds.
 * Provides functions to manage:
 *  * Owner (set/get/enforce)
 *  * Adding, replacing, removing function selectors
 *  * Running the diamondCut process
 *
 * Diamond Storage
 * - Mapping from function selector to facet address
 * - Mapping from facet to its selectors
 * - Array of all facet addresses
 * - Owner
 * - Supported interfaces (ERC-165)
 *
 * Contract Owner
 * - Only the owner can call `diamondCut`.
 *
 * diamondCut
 * - Takes an array of facet changes
 * - Updates storage routing table accordingly
 * - Optionally calls an init function
 */

library LibSmartSolve {
    // unique slot for DiamondStorage
    bytes32 internal constant DIAMOND_STORAGE_POSITION =
        keccak256("smart-solve.diamond.storage");

    // ---------------- Data Structures ----------------


    /// @notice Holds info about one function selector
    /// @param facetAddress The facet that implements this selector
    /// @param selectorPosition The index of this selector inside that facet’s selector array
    struct FacetAddressAndSelectorPosition {
        address facetAddress;
        uint16 selectorPosition; // index in facetFunctionSelectors
    }

    /// @notice Holds all selectors for a given facet
    /// @param selectors The array of function selectors for this facet
    /// @param facetAddressPosition The index of this facet in the global facetAddresses array
    struct FacetFunctionSelectors {
        bytes4[] selectors; // functions this facet implements
        uint16 facetAddressPosition; // index in facetAddresses array
    }


    /// @notice Global storage for the diamond
    /// @dev Lives at DIAMOND_STORAGE_POSITION slot
    struct DiamondStorage {
        mapping(bytes4 => FacetAddressAndSelectorPosition) selectorToFacetAndPosition;
        mapping(address => FacetFunctionSelectors) facetFunctionSelectors;
        address[] facetAddresses;
        address contractOwner;
        mapping(bytes4 => bool) supportedInterfaces; // for ERC-165
    }

    // ---------------- Access to Storage ----------------

    /// @notice Get pointer to diamond storage
    /// @return ds Reference to DiamondStorage struct
    function diamondStorage() internal pure returns (DiamondStorage storage ds) {
        bytes32 position = DIAMOND_STORAGE_POSITION;
        assembly {
            ds.slot := position
        }
    }

    // ---------------- Ownership ----------------

    /// @notice Emitted when ownership changes
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    /// @notice Sets the contract owner
    /// @param newOwner The address of the new owner
    function setContractOwner(address newOwner) internal {
        DiamondStorage storage ds = diamondStorage();
        address prevOwner = ds.contractOwner;
        ds.contractOwner = newOwner;
        emit OwnershipTransferred(prevOwner, newOwner);
    }

    /// @notice Returns the current contract owner
    /// @return The address of the contract owner
    function contractOwner() internal view returns (address) {
        return diamondStorage().contractOwner;
    }

    /// @notice Ensures the caller is the contract owner
    /// @dev Reverts if msg.sender != contractOwner
    function enforceIsContractOwner() internal view {
        require(msg.sender == diamondStorage().contractOwner, "LibSmartSolve: Must be contract owner");
    }

    // ---------------- DiamondCut Logic ----------------

    /// @notice Executes a diamond cut (add/replace/remove functions)
    /// @param _diamondCut Array of facet changes
    /// @param _init Address of contract to call after upgrade (optional)
    /// @param _calldata Encoded function call for _init (optional)
    function diamondCut(
        IDiamondCut.FacetCut[] memory _diamondCut,
        address _init,
        bytes memory _calldata
    ) internal {
        for (uint256 i; i < _diamondCut.length; i++) {
            IDiamondCut.FacetCutAction action = _diamondCut[i].action;
            if (action == IDiamondCut.FacetCutAction.Add) {
                addFunctions(_diamondCut[i].facetAddress, _diamondCut[i].functionSelectors);
            } else if (action == IDiamondCut.FacetCutAction.Replace) {
                replaceFunctions(_diamondCut[i].facetAddress, _diamondCut[i].functionSelectors);
            } else if (action == IDiamondCut.FacetCutAction.Remove) {
                removeFunctions(_diamondCut[i].facetAddress, _diamondCut[i].functionSelectors);
            } else {
                revert("LibSmartSolve: Incorrect FacetCutAction");
            }
        }
        emit IDiamondCut.DiamondCut(_diamondCut, _init, _calldata);
        initializeDiamondCut(_init, _calldata);
    }

    // ---------------- Helpers for DiamondCut ----------------

    /// @notice Adds new function selectors and maps them to a facet
    /// @param _facet Address of the facet providing these functions
    /// @param _selectors List of function selectors to add
    function addFunctions(address _facet, bytes4[] memory _selectors) internal {
        require(_facet != address(0), "LibSmartSolve: facet address is zero");
        DiamondStorage storage ds = diamondStorage();

        // if facet is new, push it to facetAddresses
        if (ds.facetFunctionSelectors[_facet].selectors.length == 0) {
            ds.facetFunctionSelectors[_facet].facetAddressPosition = uint16(ds.facetAddresses.length);
            ds.facetAddresses.push(_facet);
        }

        for (uint256 i; i < _selectors.length; i++) {
            bytes4 selector = _selectors[i];
            require(ds.selectorToFacetAndPosition[selector].facetAddress == address(0),
                "LibSmartSolve: selector already exists");
            ds.facetFunctionSelectors[_facet].selectors.push(selector);
            ds.selectorToFacetAndPosition[selector] =
                FacetAddressAndSelectorPosition(_facet, uint16(ds.facetFunctionSelectors[_facet].selectors.length - 1));
        }
    }

    /// @notice Replaces existing selectors with ones from a new facet
    /// @param _facet Address of the new facet
    /// @param _selectors Function selectors to replace
    function replaceFunctions(address _facet, bytes4[] memory _selectors) internal {
        require(_facet != address(0), "LibSmartSolve: facet address is zero");
        for (uint256 i; i < _selectors.length; i++) {
            removeFunction(_selectors[i]);
            addFunctions(_facet, toSingletonArray(_selectors[i]));
        }
    }

    /// @notice Removes function selectors from the diamond
    /// @param _facet Ignored (kept for event compatibility)
    /// @param _selectors Function selectors to remove
    function removeFunctions(address _facet, bytes4[] memory _selectors) internal {
        // _facet param is unused but kept for event consistency
        for (uint256 i; i < _selectors.length; i++) {
            removeFunction(_selectors[i]);
        }
    }

    function removeFunction(bytes4 _selector) private {
        DiamondStorage storage ds = diamondStorage();
        FacetAddressAndSelectorPosition memory old = ds.selectorToFacetAndPosition[_selector];
        require(old.facetAddress != address(0), "LibSmartSolve: selector does not exist");

        // get facet selectors array
        bytes4[] storage selectors = ds.facetFunctionSelectors[old.facetAddress].selectors;
        uint256 lastPos = selectors.length - 1;
        bytes4 lastSelector = selectors[lastPos];

        // swap & pop
        selectors[old.selectorPosition] = lastSelector;
        ds.selectorToFacetAndPosition[lastSelector].selectorPosition = old.selectorPosition;
        selectors.pop();

        // if facet now empty, remove facet address
        if (selectors.length == 0) {
            uint16 lastAddrPos = uint16(ds.facetAddresses.length - 1);
            address lastAddr = ds.facetAddresses[lastAddrPos];
            ds.facetAddresses[ds.facetFunctionSelectors[old.facetAddress].facetAddressPosition] = lastAddr;
            ds.facetFunctionSelectors[lastAddr].facetAddressPosition =
                ds.facetFunctionSelectors[old.facetAddress].facetAddressPosition;
            ds.facetAddresses.pop();
            delete ds.facetFunctionSelectors[old.facetAddress];
        }

        delete ds.selectorToFacetAndPosition[_selector];
    }

    // ---------------- Init Call ----------------


    /// @notice Executes optional init call after diamondCut
    /// @param _init The contract to delegatecall
    /// @param _calldata Encoded function call for _init
    /// @dev Allows running setup logic (e.g., initializing storage) after upgrade
    function initializeDiamondCut(address _init, bytes memory _calldata) private {
        if (_init == address(0)) {
            require(_calldata.length == 0, "LibSmartSolve: _init is zero but calldata is not empty");
        } else {
            (bool success, bytes memory error) = _init.delegatecall(_calldata);
            if (!success) {
                if (error.length > 0) {
                    // bubble up error
                    assembly {
                        revert(add(error, 32), mload(error))
                    }
                } else {
                    revert("LibSmartSolve: _init function reverted");
                }
            }
        }
    }

    // ---------------- Utility ----------------

    /// @notice Helper to pack a single selector into an array
    /// useful for single function operations
    /// @param selector The function selector
    /// @return arr A new array containing only that selector
    /// @dev Useful when replacing a single function
    function toSingletonArray(bytes4 selector) private pure returns (bytes4[] memory arr) {
        arr = new bytes4[](1);
        arr[0] = selector;
    }
}
